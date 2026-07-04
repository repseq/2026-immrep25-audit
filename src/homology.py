"""Homology test — within- vs between-epitope CDR3 near-neighbour enrichment.

For a single chain, group TCRs by epitope. Only equal-length CDR3s can be at a
given Hamming distance, so all pair counting is done *within* a length bin.

Definitions (per cohort, per chain), matching the corrected normalization:
    P_self    = sum_e C(N_e, 2)          over equal-length pairs   (within-epitope)
    P_nonself = sum_L C(N_L,2) - P_self   over equal-length pairs   (cross-epitope)
    m(<=d)    = within-epitope pairs at Hamming <= d
    l(<=d)    = cross-epitope  pairs at Hamming <= d
    S/N(d)    = (m/P_self) / (l/P_nonself)

d = 0 isolates *publicity* (identical CDR3s); d = 1 is the Dash-et-al. convergence
signal; d = 2,3 show how S/N grows with substitution radius. Signal -> S/N >> 1,
noise -> S/N ~ 1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_AA_IDX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def _encode(seq: str) -> np.ndarray:
    return np.fromiter((_AA_IDX.get(c, 20) for c in seq), dtype=np.int8, count=len(seq))


def count_pairs(seqs: np.ndarray, epi: np.ndarray, max_d: int = 3) -> dict:
    """Exact self/non-self pair counts by Hamming distance, summed over length bins.

    Returns dict with cumulative m[d], l[d] (d=0..max_d) and denominators P_self, P_nonself.
    """
    m = np.zeros(max_d + 1)   # within-epitope pairs at Hamming == d
    l = np.zeros(max_d + 1)   # cross-epitope pairs at Hamming == d
    P_self = 0.0
    P_nonself = 0.0
    lengths = np.array([len(s) for s in seqs])
    for L in np.unique(lengths):
        idx = np.where(lengths == L)[0]
        if idx.size < 2:
            continue
        mat = np.stack([_encode(seqs[i]) for i in idx])          # (n, L) int8
        e = epi[idx]
        n = idx.size
        # pairwise Hamming (upper triangle)
        iu, ju = np.triu_indices(n, k=1)
        d = (mat[iu] != mat[ju]).sum(axis=1)                     # (n_pairs,)
        same_epi = e[iu] == e[ju]
        # denominators (equal-length pairs)
        tot = iu.size
        _, cnts = np.unique(e, return_counts=True)
        self_pairs = float((cnts * (cnts - 1) // 2).sum())
        P_self += self_pairs
        P_nonself += tot - self_pairs
        # tally by exact distance
        for dd in range(0, max_d + 1):
            sel = d == dd
            if not sel.any():
                continue
            m[dd] += np.count_nonzero(sel & same_epi)
            l[dd] += np.count_nonzero(sel & ~same_epi)
    return {"m_exact": m, "l_exact": l, "P_self": P_self, "P_nonself": P_nonself}


def sn_from_counts(c: dict, max_d: int = 3, pseudo: float = 0.5) -> pd.DataFrame:
    """Cumulative S/N(<=d) from exact counts.

    A Haldane continuity correction (add `pseudo` to matched-pair counts) keeps
    S/N finite when a cohort has zero cross-epitope neighbours (perfect
    separation, i.e. S/N -> infinity) — a strong-signal case, not missing data.
    """
    m_cum = np.cumsum(c["m_exact"])
    l_cum = np.cumsum(c["l_exact"])
    ps, pn = c["P_self"], c["P_nonself"]
    rows = []
    for d in range(0, max_d + 1):
        r_self = m_cum[d] / ps if ps else np.nan
        r_non = l_cum[d] / pn if pn else np.nan
        r_self_c = (m_cum[d] + pseudo) / ps if ps else np.nan
        r_non_c = (l_cum[d] + pseudo) / pn if pn else np.nan
        sn = (r_self_c / r_non_c) if (r_non_c and r_non_c > 0) else np.nan
        rows.append({"d": d, "m": m_cum[d], "l": l_cum[d],
                     "r_self": r_self, "r_nonself": r_non, "sn": sn})
    return pd.DataFrame(rows)


def _cross_hist_by_len(seqs_e, seqs_o, max_d: int, block: int = 200):
    """Exact-distance histogram counts[0..max_d] of cross pairs (equal length) + denom."""
    seqs_e = list(seqs_e); seqs_o = list(seqs_o)
    le = np.array([len(s) for s in seqs_e]); lo = np.array([len(s) for s in seqs_o])
    hist = np.zeros(max_d + 1); denom = 0.0
    for L in np.unique(le):
        ie = np.where(le == L)[0]; io = np.where(lo == L)[0]
        if io.size == 0:
            continue
        Ao = np.stack([_encode(seqs_o[i]) for i in io])
        denom += ie.size * io.size
        for s in range(0, ie.size, block):
            blk = np.stack([_encode(seqs_e[i]) for i in ie[s:s + block]])
            dist = (blk[:, None, :] != Ao[None, :, :]).sum(axis=2)
            for dd in range(max_d + 1):
                hist[dd] += float(np.count_nonzero(dist == dd))
    return hist, denom


def per_epitope_sn(long_chain: pd.DataFrame, max_d: int = 3, min_n: int = 30,
                   max_e: int = 300, max_rest: int = 4000, pseudo: float = 0.5,
                   seed: int = 0) -> pd.DataFrame:
    """One-vs-many S/N of each epitope: within-epitope vs epitope-vs-rest neighbours.

    One row per epitope with >= min_n records; columns m<d>, l<d>, sn<d> (cumulative
    Hamming <= d) for d in 1..max_d, plus the d=1 within-epitope neighbour rate.
    Large sets are capped (epitope to max_e, rest to max_rest) for tractability.
    """
    rng = np.random.default_rng(seed)
    vc = long_chain.epitope.value_counts()
    eps = vc[vc >= min_n].index.to_numpy()
    rows = []
    for e in eps:
        E = long_chain[long_chain.epitope == e]
        if len(E) > max_e:
            E = E.sample(max_e, random_state=int(rng.integers(1 << 31)))
        R = long_chain[long_chain.epitope != e]
        if len(R) > max_rest:
            R = R.sample(max_rest, random_state=int(rng.integers(1 << 31)))
        cw = count_pairs(E.cdr3.to_numpy(), np.array(["x"] * len(E)), max_d=max_d)
        m_cum = np.cumsum(cw["m_exact"]); P_self = cw["P_self"]
        lhist, P_non = _cross_hist_by_len(E.cdr3.to_numpy(), R.cdr3.to_numpy(), max_d)
        l_cum = np.cumsum(lhist)
        row = {"epitope": e, "n": int(vc[e]), "P_self": P_self, "P_non": P_non}
        for d in range(1, max_d + 1):
            row["m%d" % d] = m_cum[d]; row["l%d" % d] = l_cum[d]
            # exposure-proportional pseudocount: S/N -> 1 (not P_non/P_self) when counts are 0
            if P_self and P_non:
                r_self = (m_cum[d] + pseudo) / P_self
                r_non = (l_cum[d] + pseudo * P_non / P_self) / P_non
                row["sn%d" % d] = r_self / r_non
            else:
                row["sn%d" % d] = np.nan
        row["r_self"] = m_cum[1] / P_self if P_self else np.nan
        row["r_nonself"] = l_cum[1] / P_non if P_non else np.nan
        rows.append(row)
    cols = (["epitope", "n", "P_self", "P_non"]
            + [c % d for d in range(1, max_d + 1) for c in ("m%d", "l%d", "sn%d")]
            + ["r_self", "r_nonself"])
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("m1", ascending=False).reset_index(drop=True)


def dataset_sn(per_ep: pd.DataFrame, d: int = 1):
    """Geometric mean +/- 95% CI (log10 SE) of per-epitope S/N across a cohort."""
    s = per_ep["sn%d" % d].values
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) == 0:
        return dict(sn=np.nan, lo=np.nan, hi=np.nan, n_ep=0)
    lg = np.log10(s)
    mu = lg.mean()
    se = lg.std(ddof=1) / np.sqrt(len(lg)) if len(lg) > 1 else 0.0
    return dict(sn=10 ** mu, lo=10 ** (mu - 1.96 * se), hi=10 ** (mu + 1.96 * se), n_ep=len(s))


def _subsample(long_chain: pd.DataFrame, K: int, E_cap: int, rng) -> pd.DataFrame:
    """Sample E_cap epitopes each with >= K rows, then K rows per epitope."""
    vc = long_chain.epitope.value_counts()
    eligible = vc[vc >= K].index.to_numpy()
    if eligible.size == 0:                       # fall back: keep all, cap by K
        eligible = vc.index.to_numpy()
    if eligible.size > E_cap:
        eligible = rng.choice(eligible, E_cap, replace=False)
    parts = []
    for ep in eligible:
        sub = long_chain[long_chain.epitope == ep]
        k = min(K, len(sub))
        parts.append(sub.sample(k, replace=False, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts, ignore_index=True)


def homology_sn_bootstrap(long_chain: pd.DataFrame, K: int = 30, E_cap: int = 30,
                          n_boot: int = 200, max_d: int = 3, seed: int = 0) -> pd.DataFrame:
    """Bootstrap matched-design S/N(<=d). Returns per-boot rows (long form)."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_boot):
        s = _subsample(long_chain, K, E_cap, rng)
        c = count_pairs(s.cdr3.to_numpy(), s.epitope.to_numpy(), max_d)
        sn = sn_from_counts(c, max_d)
        sn["boot"] = b
        sn["n_epitopes"] = s.epitope.nunique()
        out.append(sn)
    return pd.concat(out, ignore_index=True)


def summarize(boot: pd.DataFrame) -> pd.DataFrame:
    """Median and 95% CI of S/N per distance across bootstraps."""
    g = boot.groupby("d")
    return pd.DataFrame({
        "d": g.size().index,
        "sn_median": g["sn"].median().values,
        "sn_lo": g["sn"].quantile(0.025).values,
        "sn_hi": g["sn"].quantile(0.975).values,
        "r_self": g["r_self"].median().values,
        "r_nonself": g["r_nonself"].median().values,
        "n_epitopes": g["n_epitopes"].median().values,
    })


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from cohorts import build_cohorts, HIERARCHY, COHORT_META
    coh = build_cohorts()
    for chain in ("B", "A"):
        print(f"\n==== chain TR{chain}  (K=30, matched, n_boot=80) ====")
        for name in HIERARCHY:
            if name not in coh:
                continue
            lc = coh[name]["long"]
            lc = lc[lc.chain == chain]
            boot = homology_sn_bootstrap(lc, K=30, E_cap=30, n_boot=80, max_d=3, seed=1)
            s = summarize(boot)
            r = s[s.d == 1].iloc[0]
            print(f"  {COHORT_META[name]['label']:<13} d1 S/N={r.sn_median:6.2f} "
                  f"[{r.sn_lo:5.2f},{r.sn_hi:6.2f}]  (E={r.n_epitopes:.0f})")
