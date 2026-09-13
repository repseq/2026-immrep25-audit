"""Leiden clustering of ESM-2 embeddings, scored by standard external-validation metrics.

A clustering-native, model-based probe of epitope structure. For each cohort we build a kNN
cosine graph on the cached ESM-2 embeddings, partition it with Leiden, and score the
partition against the epitope labels with two community-standard, chance-corrected external
clustering-validation metrics:

  * Adjusted Rand Index (ARI): the Rand index (fraction of TCR pairs placed concordantly by
    the clustering and the epitope labeling) corrected for chance; 0 = random, 1 = perfect.
  * Adjusted Mutual Information (AMI): the mutual information between partition and labels,
    corrected for chance and normalized to [0, 1].

Both are single global scalars per cohort, so an 80%-subsample bootstrap over TCRs puts a CI
on them and gives a direct IMMREP22-versus-IMMREP25 test. Reuses the embeddings cached by
esm_signal.py (cache/esm/*.npy); no re-embedding.

ARI is corrected for chance but NOT for the size of the comparison: with s epitopes and r < s
Leiden communities, ARI = 1 is unreachable, and the attainable maximum falls as s grows (for the
deepest cohort here, s = 83 against r = 10, it is 0.20 against 0.52 for a 20-epitope cohort). The
minimum is not the problem -- by Chacon & Rastrojo (Adv. Data Anal. Classif. 17:125-133, 2023,
Theorem 1) it is about -0.02 at these sizes. The maximum over ALL partitions with the observed
marginals has no closed form (Hubert & Arabie 1985), but over the partitions the probe could
actually return -- r groups, no epitope split -- it does: ARI_max = 2(r-1)/(r+s-2), derived in
_ari_ceiling. We report ARI as a percentage of that, which puts every cohort on one 0-100 scale
alongside AMI, which is normalized and needs no such rescaling.

Usage: python src/esm_cluster.py   # writes results/esm_cluster.csv + esm_cluster_{A,B}.dat
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META          # noqa: E402
from esm_signal import _subsample, embed, CACHE, K_NN               # noqa: E402
from mi import mi_report                                            # noqa: E402

import igraph as ig                                                 # noqa: E402
import leidenalg                                                    # noqa: E402
from sklearn.neighbors import NearestNeighbors                      # noqa: E402
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score  # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
KNN = K_NN        # neighbours for the Leiden graph -- the same k as the purity statistic
RES = 1.0         # Leiden resolution (RBConfiguration)
N_BOOT = 1000     # bootstrap resamples (with replacement) for the metric CIs
N_PERM = 1000     # permutation null for MI-in-bits


def _mn_ci(point, boot):
    """Percentile bootstrap CI: recentre the (with-replacement) resample deviations on the point."""
    boot = np.asarray(boot)
    if not len(boot):
        return (np.nan, np.nan)
    dev = boot - boot.mean()
    return float(point + np.percentile(dev, 2.5)), float(point + np.percentile(dev, 97.5))


def _ari_ceiling(s: int, r: int) -> float:
    """Highest ARI an r-cluster partition can reach against s equal-sized epitopes: 2(r-1)/(r+s-2).

    The denominator ARI is reported as a percentage of, and the only thing about a cohort it needs
    is its epitope count. Derivation, for s epitopes of m = n/s receptors and r <= s groups that
    merge whole epitopes into equal totals n/r (equal totals minimise sum_g S_g^2 and so maximise
    ARI), all pair counts to leading order in n:
        same-cluster same-epitope   a     = s*C(m,2)     = n^2/(2s)
        same-cluster                a + c = r*C(n/r,2)   = n^2/(2r)
        same-epitope                B     = a            = n^2/(2s)
        all pairs                   N     = C(n,2)       = n^2/2
    so the chance term is (a+c)B/N = n^2/(2rs) and, dividing through by n^2/2,
        ARI_max = [1/s - 1/(rs)] / [(1/r + 1/s)/2 - 1/(rs)] = 2(r-1)/(r + s - 2).
    r >= s gives 1 and r = 1 gives 0, as they must. Note what does NOT scale with s: a random
    partition scores ARI = 0 for every s, which is what the chance correction buys. It is the
    CEILING that carries the size penalty -- ~2(r-1)/s once s >> r -- so dividing by it is what
    makes cohorts of different depth comparable.

    The relaxation is that groups of exactly n/r cannot always be built out of whole epitopes, so
    this is an UPPER bound on the attainable maximum, and the pair counts above drop an O(1/m) term.
    Both errors run the same way: on the shapes reported here the form is 0-8% high (worst near
    s/r = 1.5, ~2% by s/r = 8, and exact when r >= s), so every reported percentage is slightly
    conservative, and conservative for every cohort in the same direction.
    demo() checks the form against the exact whole-epitope maximum on the real cohort shapes.
    """
    g = min(int(r), int(s))
    return 0.0 if g < 2 else 2.0 * (g - 1) / (g + int(s) - 2)


def _ari_ceiling_exact(sizes: np.ndarray, r: int) -> float:
    """Size-aware version of _ari_ceiling, used only to check it (demo()).

    With r groups and epitope sizes n_c, merging is forced whenever r < s, and every merged pair of
    epitopes contributes a same-cluster/different-epitope pair; ARI is maximised by making the group
    totals as equal as possible, which longest-processing-time greedy achieves.
    """
    def pairs(x):
        return x * (x - 1) / 2.0
    g = min(int(r), len(sizes))
    tot = np.zeros(g)
    for sz in np.sort(sizes)[::-1]:
        tot[np.argmin(tot)] += sz
    n = sizes.sum()
    a = pairs(sizes).sum()                       # same cluster and same epitope
    c = pairs(tot).sum() - a                     # same cluster, different epitope
    d = pairs(n) - a - c                         # b = 0: no epitope is split
    N = a + c + d
    e = a * (a + c) + (c + d) * d                # b=0 collapses (a+b)(a+c) and (c+d)(b+d)
    return float((N * (a + d) - e) / (N ** 2 - e))


def demo():
    """The closed form must bound the exact whole-epitope maximum, tightly, on reported shapes."""
    assert _ari_ceiling(20, 20) == 1.0 and _ari_ceiling(20, 40) == 1.0    # r >= s
    assert _ari_ceiling(20, 1) == 0.0                                      # one cluster
    worst = 0.0
    for s, r in ((2, 5), (7, 6), (7, 7), (12, 8), (20, 8), (20, 9), (20, 10), (20, 11),
                 (31, 13), (46, 13), (50, 8), (83, 10)):
        e = _ari_ceiling_exact(np.full(s, 45.0), r)   # MIN_N=30..PER_EPI=60, midpoint
        dev = (_ari_ceiling(s, r) - e) / e
        assert dev >= -1e-12, (s, r, dev)             # an upper bound, never below
        worst = max(worst, dev)
    assert worst < 0.09, worst
    print("esm_cluster.demo OK  (closed-form ARI ceiling <= exact + %.1f%%)" % (100 * worst))


def _partition(X: np.ndarray, seed: int = 0) -> np.ndarray:
    n = len(X)
    nn = NearestNeighbors(n_neighbors=min(KNN + 1, n), metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    edges = [(i, int(j)) for i, row in enumerate(idx[:, 1:]) for j in row]
    g = ig.Graph(n=n, edges=edges, directed=False); g.simplify()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=RES, seed=seed)
    return np.asarray(part.membership)


def _scores(X: np.ndarray, yt: np.ndarray, seed: int = 0) -> dict:
    pred = _partition(X, seed=seed)
    return dict(n_clusters=len(set(pred)),
                ari=float(adjusted_rand_score(yt, pred)),
                ami=float(adjusted_mutual_info_score(yt, pred)))


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "=\\mathrm{n/a}"
    # _p() floors the estimate at 1/B = 1e-3, the bootstrap's resolution limit, so a value
    # sitting on that floor means "below what B resamples can resolve" -- report it as such
    # rather than as an exact equality.
    if p <= 1e-3:
        return "<0.001"
    if p < 1e-2:
        return "=%.3f" % p
    return "=%.2f" % p


def run():
    rng = np.random.default_rng(0)
    coh = build_cohorts(include_olga=True)
    order = [c for c in HIERARCHY if c in coh]
    rows = []
    for name in order:
        long = coh[name]["long"]
        for chain in ("A", "B"):
            if name == "mlr_prolif" and chain == "A":
                continue
            sub = _subsample(long[long.chain == chain][["epitope", "cdr3"]].dropna(), rng)
            if sub.epitope.nunique() < 2:
                continue
            cache = os.path.join(CACHE, f"{name}_{chain}.npy")
            seqs = sub.cdr3.to_list()
            if os.path.exists(cache) and np.load(cache).shape[0] == len(seqs):
                X = np.load(cache)
            else:
                X = embed(seqs); np.save(cache, X)
            yt = pd.Categorical(sub.epitope).codes.astype(np.int64)
            base = _scores(X, yt, seed=0)
            n_ep = int(sub.epitope.nunique())
            mi = mi_report(yt, _partition(X, seed=0), n_perm=N_PERM)   # I(epitope; cluster) in bits
            ab, mb = [], []                                    # bootstrap (n with replacement)
            for b in range(N_BOOT):
                s = rng.choice(len(X), len(X), replace=True)
                if len(set(yt[s])) < 2:
                    continue
                sc = _scores(X[s], yt[s], seed=b)
                ab.append(sc["ari"]); mb.append(sc["ami"])
            ab, mb = np.asarray(ab), np.asarray(mb)
            al, ah = _mn_ci(base["ari"], ab); ml, mh = _mn_ci(base["ami"], mb)
            # ARI has no correction for the size of the comparison: a cohort whose receptors carry
            # s epitopes but whose graph partitions into r < s communities cannot reach ARI = 1 at
            # all, so ARI falls as s grows even when the embedding is unchanged. We therefore also
            # report ARI as a percentage of the maximum this partition size could attain. The
            # ceiling is a constant per cohort, so the interval scales with it.
            ceil = _ari_ceiling(n_ep, base["n_clusters"])
            rows.append({"cohort": name, "label": COHORT_META[name]["label"], "chain": chain,
                         "n": len(X), "n_ep": n_ep, "ari_ceiling": ceil,
                         "K": KNN, "resolution": RES, **base,
                         "mi_bits": mi["excess"], "mi_U": mi["U"], "mi_p": mi["p"],
                         "ari_lo": al, "ari_hi": ah, "ami_lo": ml, "ami_hi": mh,
                         "ari_pct": 100 * base["ari"] / ceil, "ari_pct_lo": 100 * al / ceil,
                         "ari_pct_hi": 100 * ah / ceil,
                         "_ariboot": ab, "_amiboot": mb, "_aripctboot": 100 * ab / ceil})
            print("%-16s TR%s  ARI=%.3f [%.3f,%.3f]  =%.1f%% of max %.2f  AMI=%.3f"
                  "  MI=%.3f bits(U=%.3f)  clusters=%d ep=%d n=%d" %
                  (name, chain, base["ari"], al, ah, 100 * base["ari"] / ceil, ceil, base["ami"],
                   mi["excess"], mi["U"], base["n_clusters"], n_ep, len(X)))
    df = pd.DataFrame(rows)

    def _boot(name, chain, col):
        r = df[(df.cohort == name) & (df.chain == chain)]
        return r[col].iloc[0] if len(r) else np.array([])

    def _p(a, b):              # two-sided difference test; independent resamples paired by permutation
        if not (len(a) and len(b)):
            return np.nan
        k = min(len(a), len(b))
        d = rng.permutation(a)[:k] - rng.permutation(b)[:k]
        p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        return max(p, 1.0 / k)                                  # precision capped at 1/B (M9)

    # (column stem, macro stem, bootstrap column). ARI-as-%-of-max gets its own
    # IMMREP22-vs-IMMREP25 test: the two cohorts carry different ceilings, so their scalings differ
    # and the test cannot be inherited from the unscaled ARI.
    METRICS = (("ari", "ari", "_ariboot", "%.3f"), ("ami", "ami", "_amiboot", "%.3f"),
               ("ari_pct", "aripct", "_aripctboot", "%.1f"))
    macros = {}                                               # letter-only names (LaTeX macros)
    for chain in ("B", "A"):
        for col_stem, mac, boot, fmt in METRICS:
            p = _p(_boot("immrep22_true", chain, boot), _boot("immrep25_pos", chain, boot))
            macros["%sP%s" % (mac, chain)] = _fmt_p(p)
            for who, key in (("immrep22_true", "Ii"), ("immrep25_pos", "Imm"),
                             ("vdjdb_hq", "Hq"), ("vdjdb_lq", "Lq"), ("tcrvdb_true", "Tt"),
                             ("pairseq_mock", "Ps"), ("airr_control", "Airr")):
                r = df[(df.cohort == who) & (df.chain == chain)]
                if len(r):
                    macros["%s%s%s" % (mac, key, chain)] = fmt % r[col_stem].iloc[0]
                    macros["%sLo%s%s" % (mac, key, chain)] = fmt % r[col_stem + "_lo"].iloc[0]
                    macros["%sHi%s%s" % (mac, key, chain)] = fmt % r[col_stem + "_hi"].iloc[0]
                    if col_stem == "ami":
                        macros["esmMiBits%s%s" % (key, chain)] = "%.3f" % r["mi_bits"].iloc[0]
                        macros["esmMiU%s%s" % (key, chain)] = "%.3f" % r["mi_U"].iloc[0]
                        macros["esmNep%s%s" % (key, chain)] = "%d" % r["n_ep"].iloc[0]
                        macros["esmClust%s%s" % (key, chain)] = "%d" % r["n_clusters"].iloc[0]
                        macros["esmCeil%s%s" % (key, chain)] = "%.2f" % r["ari_ceiling"].iloc[0]
        print("  [imm22 vs imm25 TR%s]  ARI %s vs %s  p%s ;  AMI p%s ;  ARI%% of max %s vs %s  p%s"
              % (chain, macros.get("ariIi" + chain), macros.get("ariImm" + chain),
                 macros["ariP" + chain], macros["amiP" + chain],
                 macros.get("aripctIi" + chain), macros.get("aripctImm" + chain),
                 macros["aripctP" + chain]))

    # The bootstrap arrays are the only inputs from which an equivalence bound (largest effect
    # ruled out) can be computed, and this stage takes ~25 min because the bootstrap refits
    # Leiden N_BOOT times per cohort and chain. Persist them before dropping, so a bound can be
    # recomputed in seconds instead of re-running the stage.
    np.savez_compressed(
        os.path.join(CACHE, "boot_arrays.npz"),
        **{"%s|%s|%s" % (r.cohort, r.chain, k): np.asarray(getattr(r, k), dtype=float)
           for r in df.itertuples() for k in ("_ariboot", "_amiboot", "_aripctboot")})
    print("  [wrote %s: %d bootstrap arrays, B=%d]"
          % (os.path.join(CACHE, "boot_arrays.npz"), 3 * len(df), N_BOOT))

    out = df.drop(columns=["_ariboot", "_amiboot", "_aripctboot"])
    out.to_csv(os.path.join(RESULTS, "esm_cluster.csv"), index=False)
    idx = {n: i for i, n in enumerate(HIERARCHY)}
    for chain in ("B", "A"):
        b = out[out.chain == chain].copy(); b["x"] = [idx[c] for c in b.cohort]
        with open(os.path.join(ADAT, f"esm_cluster_{chain}.dat"), "w") as fh:
            fh.write("# x cohort ari ari_lo ari_hi ami ami_lo ami_hi"
                     " ari_pct ari_pct_lo ari_pct_hi\n")
            b[["x", "cohort", "ari", "ari_lo", "ari_hi", "ami", "ami_lo", "ami_hi",
               "ari_pct", "ari_pct_lo", "ari_pct_hi"]].to_csv(fh, sep=" ", index=False,
                                                             header=False)
    with open(os.path.join(ADAT, "esm_cluster_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/esm_cluster.csv + esm_cluster_{A,B}.dat + esm_cluster_macros.tex")


if __name__ == "__main__":
    demo()
    run()
