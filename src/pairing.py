"""Pairing test — is V/J usage non-random *given the epitope*?

Two complementary signals (ref PMID 32163410), each scored against a permutation
null so all cohorts sit on one dimensionless scale:

  gene-usage bias : mean_e KL( P_e(gene) || Q(gene) ) over axes Va,Ja,Vb,Jb,
                    where P_e is the within-epitope gene distribution and Q the
                    cohort background. Null = shuffle epitope labels.
  inter-chain MI  : mean_e I( (Va,Ja) ; (Vb,Jb) ) within epitope, Miller-Madow
                    corrected. Null = shuffle the alpha<->beta pairing within
                    each epitope (destroys coupling, keeps marginals).

Reported as z = (obs - null_mean) / null_std. Signal -> z >> 0; random -> z ~ 0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

AXES = ["va", "ja", "vb", "jb"]


def _entropy_mm(counts: np.ndarray) -> float:
    """Miller-Madow entropy (bits)."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    H = -(p * np.log2(p)).sum()
    m = (counts > 0).sum()
    return H + (m - 1) / (2 * n * np.log(2))


def _mi_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Miller-Madow mutual information (bits) between two integer-coded vectors."""
    if len(a) < 2:
        return 0.0
    ca = np.bincount(a); cb = np.bincount(b)
    joint = a.astype(np.int64) * (b.max() + 1) + b
    cj = np.bincount(joint)
    return max(_entropy_mm(ca) + _entropy_mm(cb) - _entropy_mm(cj), 0.0)


def _codes(s: pd.Series) -> np.ndarray:
    return pd.Categorical(s.astype(str)).codes.astype(np.int64)


def _kl(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 0.5) -> float:
    """KL(P||Q) in bits with add-eps smoothing over a shared vocabulary."""
    p = (p_counts + eps) / (p_counts + eps).sum()
    q = (q_counts + eps) / (q_counts + eps).sum()
    return float((p * np.log2(p / q)).sum())


def gene_bias(paired: pd.DataFrame, n_perm: int, rng) -> dict:
    """Mean within-epitope KL vs background, per axis + combined, with label-shuffle null."""
    epi = paired.epitope.values
    codes = {ax: _codes(paired[ax]) for ax in AXES}
    vocab = {ax: int(codes[ax].max()) + 1 for ax in AXES}
    eps_u, inv = np.unique(epi, return_inverse=True)
    sizes = np.bincount(inv).astype(float)
    w = sizes / sizes.sum()

    def mean_kl(labels) -> np.ndarray:
        res = np.zeros(len(AXES))
        for ai, ax in enumerate(AXES):
            c = codes[ax]
            bg = np.bincount(c, minlength=vocab[ax])
            kl_e = np.zeros(len(eps_u))
            for ei in range(len(eps_u)):
                sel = labels == ei
                pe = np.bincount(c[sel], minlength=vocab[ax])
                kl_e[ei] = _kl(pe, bg)
            res[ai] = (kl_e * w).sum()
        return res

    obs = mean_kl(inv)
    null = np.array([mean_kl(rng.permutation(inv)) for _ in range(n_perm)])
    z = (obs - null.mean(0)) / (null.std(0) + 1e-9)
    return {"axis": AXES, "kl_obs": obs, "kl_null_mean": null.mean(0),
            "kl_z": z, "kl_z_combined": float(z.mean()),
            "kl_obs_combined": float(obs.mean())}


def interchain_mi(paired: pd.DataFrame, n_perm: int, rng) -> dict:
    """Weighted within-epitope MI( (Va,Ja);(Vb,Jb) ) with pairing-shuffle null."""
    A = _codes(paired["va"].astype(str) + "|" + paired["ja"].astype(str))
    B = _codes(paired["vb"].astype(str) + "|" + paired["jb"].astype(str))
    epi = paired.epitope.values
    eps_u, inv = np.unique(epi, return_inverse=True)
    sizes = np.bincount(inv).astype(float)
    w = sizes / sizes.sum()

    def weighted_mi(Bvec) -> float:
        tot = 0.0
        for ei in range(len(eps_u)):
            sel = inv == ei
            if sel.sum() < 3:
                continue
            a = pd.Categorical(A[sel]).codes
            b = pd.Categorical(Bvec[sel]).codes
            tot += w[ei] * _mi_mm(a, b)
        return tot

    obs = weighted_mi(B)
    # null: shuffle beta within each epitope (keeps marginals, kills coupling)
    null = np.empty(n_perm)
    for p in range(n_perm):
        Bs = B.copy()
        for ei in range(len(eps_u)):
            sel = np.where(inv == ei)[0]
            Bs[sel] = B[rng.permutation(sel)]
        null[p] = weighted_mi(Bs)
    z = (obs - null.mean()) / (null.std() + 1e-9)
    return {"mi_obs": obs, "mi_null_mean": float(null.mean()),
            "mi_null_std": float(null.std()), "mi_z": float(z)}


def pairing_per_epitope(paired: pd.DataFrame, min_n: int = 30, n_null: int = 40,
                        seed: int = 0) -> pd.DataFrame:
    """One-vs-many per-epitope pairing signals, each as z vs its own permutation null.

    gene_z : (KL(P_e||Q) - null) / null_sd, null = random size-n_e draws from cohort
    mi_z   : (MI((Va,Ja);(Vb,Jb)) - null) / null_sd, null = alpha<->beta shuffle within e
    """
    rng = np.random.default_rng(seed)
    codes = {ax: _codes(paired[ax]) for ax in AXES}
    vocab = {ax: int(codes[ax].max()) + 1 for ax in AXES}
    bg = {ax: np.bincount(codes[ax], minlength=vocab[ax]) for ax in AXES}
    A = _codes(paired["va"].astype(str) + "|" + paired["ja"].astype(str))
    B = _codes(paired["vb"].astype(str) + "|" + paired["jb"].astype(str))
    N = len(paired)
    eps_u, inv = np.unique(paired.epitope.values, return_inverse=True)
    rows = []
    for ei, e in enumerate(eps_u):
        idx = np.where(inv == ei)[0]
        ne = idx.size
        if ne < min_n:
            continue
        kl_obs = np.mean([_kl(np.bincount(codes[ax][idx], minlength=vocab[ax]), bg[ax]) for ax in AXES])
        knull = np.empty(n_null)
        for k in range(n_null):
            s = rng.choice(N, ne, replace=False)
            knull[k] = np.mean([_kl(np.bincount(codes[ax][s], minlength=vocab[ax]), bg[ax]) for ax in AXES])
        a = pd.Categorical(A[idx]).codes; b = pd.Categorical(B[idx]).codes
        mi_obs = _mi_mm(a, b)
        mnull = np.array([_mi_mm(a, b[rng.permutation(ne)]) for _ in range(n_null)])
        # excess over the null, in bits (robust; -> 0 under no signal, no z-score blow-up)
        rows.append({"epitope": e, "n": ne,
                     "gene_excess": kl_obs - knull.mean(), "kl_obs": kl_obs,
                     "mi_excess": mi_obs - mnull.mean(), "mi_obs": mi_obs})
    return pd.DataFrame(rows)


def dataset_pairing(per_ep: pd.DataFrame, col: str):
    """Mean +/- 95% CI (SE) of a per-epitope pairing z across the cohort."""
    v = per_ep[col].values
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_ep=0)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return dict(mean=v.mean(), lo=v.mean() - 1.96 * se, hi=v.mean() + 1.96 * se, n_ep=len(v))


def _subsample_paired(paired: pd.DataFrame, K: int, E_cap: int, rng) -> pd.DataFrame:
    vc = paired.epitope.value_counts()
    elig = vc[vc >= K].index.to_numpy()
    if elig.size == 0:
        elig = vc.index.to_numpy()
    if elig.size > E_cap:
        elig = rng.choice(elig, E_cap, replace=False)
    parts = [paired[paired.epitope == e].sample(min(K, (paired.epitope == e).sum()),
             random_state=int(rng.integers(1 << 31))) for e in elig]
    return pd.concat(parts, ignore_index=True)


def pairing_bootstrap(paired: pd.DataFrame, K: int = 30, E_cap: int = 2,
                      n_boot: int = 40, n_perm: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        s = _subsample_paired(paired, K, E_cap, rng)
        gb = gene_bias(s, n_perm, rng)
        mi = interchain_mi(s, n_perm, rng)
        rows.append({"boot": b, "gene_bias_z": gb["kl_z_combined"],
                     "gene_bias_kl": gb["kl_obs_combined"],
                     "mi_z": mi["mi_z"], "mi_obs": mi["mi_obs"],
                     "n_epitopes": s.epitope.nunique()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from cohorts import build_cohorts, HIERARCHY, COHORT_META
    coh = build_cohorts()
    print(f"{'cohort':<14}{'geneKLz':>9}{'geneKL':>8}{'MIz':>8}{'MIobs':>8}{'E':>4}")
    for name in HIERARCHY:
        if name not in coh or coh[name]["paired"] is None:
            continue
        boot = pairing_bootstrap(coh[name]["paired"], K=30, E_cap=2, n_boot=20, n_perm=60, seed=1)
        m = boot.median(numeric_only=True)
        print(f"{COHORT_META[name]['label']:<14}{m.gene_bias_z:>9.2f}{m.gene_bias_kl:>8.3f}"
              f"{m.mi_z:>8.2f}{m.mi_obs:>8.3f}{m.n_epitopes:>4.0f}")
