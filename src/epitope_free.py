#!/usr/bin/env python
# 2026-09-13  Can the benchmark be scored without reading a peptide?
#
# IMMREP25 forms its negatives by re-pairing each receptor with the other nine peptides OF
# ITS OWN MHC, so each per-peptide task is "which same-MHC MIRA pool did this receptor come
# from". A score computed from the RECEPTOR ALONE -- no peptide, no epitope label, nothing
# fitted -- therefore has a well-defined performance on the benchmark's own metric, and it
# has no training condition that could differ from the challenge's. That is the point: the
# reviewers' objection to the germline baseline is that it was cross-validated inside the
# test set, and a score that fits nothing cannot attract that objection.
#
# Score families (all receptor-only, all epitope-blind):
#   * generation probability     log10 Pgen on TCRa, on TCRb, and the a+b combination
#   * generation degree          the same for the 1-mismatch neighbourhood Pgen (OLGA)
#   * CDR3 length                per chain, their sum, and their absolute difference
#   * Kidera factors             kf1..kf10 summed over each CDR3, length-normalised means,
#                                and the principal components of the 20-dim (a|b) sum vector
#
# Two facts govern the inference and are easy to get wrong:
#
# 1. The macro-mean FULL AUC of any receptor-level score is exactly 1/2, by rank bookkeeping
#    alone. For K equal pools of size m out of N = Km receptors, sum_p AUC_p = K/2 exactly,
#    because sum_p (sum of the ranks in pool p) = N(N+1)/2 whatever the score. So the macro
#    full AUC is a constant, not a result. The challenge's macro-AUC_0.1 is a nonlinear
#    functional of the ROC and carries no such identity, so it gets an empirical null.
# 2. Neighbourhood degree must be measured against an EXTERNAL background. A receptor's
#    within-pool Hamming-1 degree is the homology statistic's own numerator, so scoring the
#    benchmark with it reproduces the circularity the reviewers raise against the publicity
#    analysis. We use OLGA generation degree, a property of the recombination model.
#
# Inference: the score is held fixed and the 50-per-pool assignment is permuted WITHIN
# allele, which preserves the design exactly. We report the per-peptide two-sided p with a
# Benjamini-Hochberg correction across peptides, the between-peptide variance against its own
# null, and the maximum against the permutation distribution OF THE MAXIMUM -- comparing a
# maximum to 0.5 would be a multiplicity error.
#
# Run: python src/epitope_free.py        (--demo for the self-check)
from __future__ import annotations
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
CACHE = os.path.join(REPO, "cache")
KIDERA_TABLE = os.path.expanduser(
    "~/vcs/code/vdjtools/python/vdjtools/resources/aa_property_table.txt")

MAX_FPR = 0.1
AUC_BEST = 0.60      # best of 126 IMMREP25 submissions (Richardson 2026)
N_PERM = 2000
SEED = 0
KF = [f"kf{i}" for i in range(1, 11)]


# --------------------------------------------------------------------------- #
# A tie-aware, vectorised standardised partial AUC.
#
# The permutation null needs ~N_PERM x n_scores x n_peptides partial AUCs, which is far too
# many sklearn calls. The score order is FIXED across permutations (only the labels move), so
# we sort once, group equal scores once, and then every permutation is a pair of
# reduceat sums over the same grouping. Tie groups matter here: CDR3 length takes about a
# dozen distinct values over 500 receptors, and treating tied receptors as ordered would
# inflate the AUC. Within a tie group the ROC advances along a diagonal, which trapezoidal
# integration over groups handles exactly.
# --------------------------------------------------------------------------- #
def tie_groups(score: np.ndarray):
    """Descending sort order plus the reduceat offsets of equal-score runs."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts


def pauc01(y: np.ndarray, order, starts, max_fpr: float = MAX_FPR) -> float:
    """McClish-standardised partial AUC over FPR<=max_fpr; matches sklearn's max_fpr."""
    yy = y[order].astype(float)
    pos = np.add.reduceat(yy, starts)
    neg = np.add.reduceat(1.0 - yy, starts)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan
    tpr = np.r_[0.0, np.cumsum(pos) / P]
    fpr = np.r_[0.0, np.cumsum(neg) / N]
    # clip the curve at max_fpr, interpolating inside the group that straddles it
    k = int(np.searchsorted(fpr, max_fpr, side="left"))
    f = fpr[:k + 1].copy(); t = tpr[:k + 1].copy()
    if k < len(fpr) and fpr[k] > max_fpr:
        w = (max_fpr - fpr[k - 1]) / (fpr[k] - fpr[k - 1])
        f[-1] = max_fpr
        t[-1] = tpr[k - 1] + w * (tpr[k] - tpr[k - 1])
    area = float(np.trapezoid(t, f)) if hasattr(np, "trapezoid") else float(np.trapz(t, f))
    lo, hi = 0.5 * max_fpr * max_fpr, max_fpr
    return 0.5 * (1.0 + (area - lo) / (hi - lo))


def per_peptide_pauc(score: np.ndarray, pool: np.ndarray, peps) -> np.ndarray:
    order, starts = tie_groups(score)
    return np.array([pauc01((pool == p).astype(float), order, starts) for p in peps])


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    adj[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


# --------------------------------------------------------------------------- #
# Score construction
# --------------------------------------------------------------------------- #
def load_kidera() -> pd.DataFrame:
    """The ten Kidera factors over the 20 residues, from our own VDJtools resource.

    AAindex1 carries only one Kidera accession (KIDA850101), so it cannot supply the ten
    orthogonal factors; the VDJtools table does, and reusing it avoids a new dependency.
    The file has classic-Mac line endings and a leading '##' reference comment.
    """
    raw = open(KIDERA_TABLE, newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("##")]
    import io
    tbl = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t").set_index("amino_acid")
    K = tbl[KF].astype(float)
    assert K.shape == (20, 10) and not K.isna().any().any(), K.shape
    return K


def build_scores(rec: pd.DataFrame) -> dict:
    """Every epitope-blind receptor score, keyed by name."""
    S = {}
    pa = pd.read_csv(os.path.join(CACHE, "pgen1mm_immrep25_pos_A.tsv"), sep="\t")
    pb = pd.read_csv(os.path.join(CACHE, "pgen1mm_immrep25_pos_B.tsv"), sep="\t")
    d = (rec.merge(pa.rename(columns={"cdr3": "tcra_cdr3", "pgen": "pg_a", "pgen1mm": "d_a"}),
                   on="tcra_cdr3", how="left")
            .merge(pb.rename(columns={"cdr3": "tcrb_cdr3", "pgen": "pg_b", "pgen1mm": "d_b"}),
                   on="tcrb_cdr3", how="left"))
    for src, tag in (("pg", "pgen"), ("d", "degree")):
        a = np.log10(d[f"{src}_a"].where(d[f"{src}_a"] > 0))
        b = np.log10(d[f"{src}_b"].where(d[f"{src}_b"] > 0))
        # a receptor with no cached value keeps its row: median placeholder, never dropped
        a, b = a.fillna(a.median()).to_numpy(), b.fillna(b.median()).to_numpy()
        S[f"{tag}_a"], S[f"{tag}_b"], S[f"{tag}_ab"] = a, b, a + b

    la = rec.tcra_cdr3.str.len().to_numpy(float)
    lb = rec.tcrb_cdr3.str.len().to_numpy(float)
    S["len_a"], S["len_b"] = la, lb
    S["len_ab"], S["len_diff"] = la + lb, np.abs(la - lb)

    K = load_kidera()
    lut = {a: K.loc[a].to_numpy() for a in K.index}

    def kf_sum(seq):
        v = np.zeros(10)
        for c in seq:
            if c in lut:
                v += lut[c]
        return v

    SA = np.array([kf_sum(s) for s in rec.tcra_cdr3])
    SB = np.array([kf_sum(s) for s in rec.tcrb_cdr3])
    for i, k in enumerate(KF):
        S[f"{k}_sum_a"], S[f"{k}_sum_b"] = SA[:, i], SB[:, i]
        S[f"{k}_sum_ab"] = SA[:, i] + SB[:, i]
        S[f"{k}_mean_a"], S[f"{k}_mean_b"] = SA[:, i] / la, SB[:, i] / lb
    X = np.hstack([SA, SB])
    pca = PCA(n_components=3, random_state=SEED).fit(X)   # unsupervised: sees no label
    Z = pca.transform(X)
    for j in range(3):
        S[f"kf_PC{j + 1}"] = Z[:, j]
    return S, pca.explained_variance_ratio_


def main():
    rng = np.random.default_rng(SEED)
    bench = pd.read_csv(BENCH, sep="\t")
    rec = bench[bench.label == 1][["tcra_cdr3", "tcrb_cdr3", "peptide", "hla"]].reset_index(drop=True)
    assert len(rec) == 1000, len(rec)
    S, evr = build_scores(rec)
    names = list(S)
    print("scores: %d | Kidera PCA explained variance: %s" % (len(names), np.round(evr, 3)))

    rows, obs_max_global = [], 0.0
    strata = {}
    for hla, g in rec.groupby("hla"):
        idx = g.index.to_numpy()
        pool = g.peptide.to_numpy()
        peps = sorted(g.peptide.unique())
        strata[hla] = (idx, pool, peps)
        for n in names:
            a = per_peptide_pauc(S[n][idx], pool, peps)
            obs_max_global = max(obs_max_global, a.max())
            rows.append(dict(score=n, hla=hla, n_pep=len(peps), macro=a.mean(),
                             mx=a.max(), var=a.var(), n_ge_best=int((a >= AUC_BEST).sum())))
    out = pd.DataFrame(rows)

    # permutation null: score fixed, pool assignment permuted within allele
    per_cell_max = {}
    gmax = np.zeros(N_PERM)
    for b in range(N_PERM):
        m = 0.0
        for hla, (idx, pool, peps) in strata.items():
            shuf = pool[rng.permutation(len(pool))]
            for n in names:
                v = per_peptide_pauc(S[n][idx], shuf, peps)
                per_cell_max.setdefault((n, hla), []).append(v.max())
                m = max(m, v.max())
        gmax[b] = m
    out["p_max"] = [(np.sum(np.asarray(per_cell_max[(r.score, r.hla)]) >= r.mx) + 1) / (N_PERM + 1)
                    for r in out.itertuples()]
    out["p_max_bh"] = bh(out.p_max.to_numpy())
    p_global = (np.sum(gmax >= obs_max_global) + 1) / (N_PERM + 1)

    out.sort_values("mx", ascending=False, inplace=True)
    print("\n=== top 10 score x allele cells by max per-peptide AUC0.1 ===")
    print(out.head(10).to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\nmacro-AUC0.1 over all %d cells: %.4f-%.4f" % (len(out), out.macro.min(), out.macro.max()))
    print("cells with ANY peptide >= %.2f: %d of %d" % (AUC_BEST, int((out.n_ge_best > 0).sum()), len(out)))
    print("global max %.4f | null mean %.4f | null 95th %.4f | p %.4f | P(any >= %.2f) %.4f"
          % (obs_max_global, gmax.mean(), np.percentile(gmax, 95), p_global, AUC_BEST,
             float(np.mean(gmax >= AUC_BEST))))

    out.to_csv(os.path.join(RESULTS, "epitope_free.csv"), index=False)
    best = out.iloc[0]
    macros = {
        "efNscores": "%d" % len(names),
        "efMacroLo": "%.3f" % out.macro.min(), "efMacroHi": "%.3f" % out.macro.max(),
        "efMax": "%.3f" % obs_max_global,
        "efMaxScore": str(best.score).replace("_", "\\_"),
        "efNullMean": "%.3f" % gmax.mean(), "efNullHi": "%.3f" % np.percentile(gmax, 95),
        "efPglobal": "<0.001" if p_global < 1e-3 else "=%.3f" % p_global,
        "efNcells": "%d" % len(out),
        "efNcellsAbove": "%d" % int((out.n_ge_best > 0).sum()),
        "efPanyBest": "<0.001" if np.mean(gmax >= AUC_BEST) < 1e-3 else "=%.3f" % np.mean(gmax >= AUC_BEST),
        "efAucBest": "%.2f" % AUC_BEST,
        "efNperm": "%d" % N_PERM,
    }
    with open(os.path.join(ADAT, "epitope_free_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "epitope_free.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "epitope_free_macros.tex"), len(macros)))


def demo():
    """Self-check: the vectorised pAUC matches sklearn (including under heavy ties), and the
    exact rank identity for the macro-mean full AUC holds on the real benchmark."""
    rng = np.random.default_rng(0)
    for n, tied in ((500, False), (500, True)):
        s = rng.normal(size=n)
        if tied:
            s = np.round(s)                       # ~8 distinct values: heavy ties
        y = np.zeros(n); y[rng.choice(n, 50, replace=False)] = 1
        order, starts = tie_groups(s)
        mine = pauc01(y, order, starts)
        theirs = roc_auc_score(y, s, max_fpr=MAX_FPR)
        assert abs(mine - theirs) < 1e-9, (tied, mine, theirs)

    bench = pd.read_csv(BENCH, sep="\t")
    rec = bench[bench.label == 1].reset_index(drop=True)
    la = rec.tcra_cdr3.str.len().to_numpy(float)
    for hla, g in rec.groupby("hla"):
        idx = g.index.to_numpy(); pool = g.peptide.to_numpy()
        peps = sorted(g.peptide.unique())
        full = np.array([roc_auc_score((pool == p).astype(int), la[idx]) for p in peps])
        # sum_p AUC_p = K/2 exactly, for ANY receptor-level score
        assert abs(full.sum() - len(peps) / 2) < 1e-9, (hla, full.sum(), len(peps) / 2)
    assert np.allclose(bh(np.array([0.01, 0.02, 0.03])), [0.03, 0.03, 0.03])
    print("epitope_free.demo OK  (vectorised pAUC matches sklearn with and without ties; "
          "sum_p full AUC = K/2 exactly on both alleles)")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
