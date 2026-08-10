"""Coarse-grained mutual information in bits, with bias correction and a permutation null.
# 2026-08-06

I(A; rep) between a discrete epitope label A and a discrete representation of a TCR
(a Leiden cluster id, a predicted-epitope label, or a gene-usage tuple). The plug-in
estimator is upward-biased at finite N; we report (i) Miller-Madow-corrected bits and,
as the robust primary, (ii) permutation-EXCESS bits = plug-in - mean(plug-in over
label-permuted nulls), with a permutation p-value. This mirrors the pairing-probe null
already used in the audit and puts the data-processing-inequality (stated in bits) on a
common, cross-representation scale.

Reused by degenerate.py (I_train vs I_heldout) and the ESM/structure AMI-in-bits table.
"""
from __future__ import annotations
import numpy as np


def _counts(a: np.ndarray, b: np.ndarray):
    ca = {v: i for i, v in enumerate(np.unique(a))}
    cb = {v: i for i, v in enumerate(np.unique(b))}
    M = np.zeros((len(ca), len(cb)))
    for x, y in zip(a, b):
        M[ca[x], cb[y]] += 1
    return M


def _mi_plugin(M: np.ndarray) -> float:
    """Plug-in MI in bits from a joint-count matrix."""
    N = M.sum()
    if N == 0:
        return 0.0
    P = M / N
    px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = P / (px @ py)
        t = P * np.log2(np.where(r > 0, r, 1.0))
    return float(np.nansum(t))


def _mi_millermadow(M: np.ndarray) -> float:
    """Miller-Madow-corrected MI in bits (correction applied to the three entropies)."""
    N = M.sum()
    if N == 0:
        return 0.0
    mx = int((M.sum(1) > 0).sum()); my = int((M.sum(0) > 0).sum()); mxy = int((M > 0).sum())
    corr_nats = (mx + my - mxy - 1) / (2.0 * N)      # H_mm additive terms collapse to this
    return _mi_plugin(M) + corr_nats / np.log(2.0)


def entropy_bits(a: np.ndarray) -> float:
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def mi_ross(X, y, k: int = 3) -> float:
    """I(X_continuous ; y_discrete) in bits -- Ross (2014) kNN estimator (as in scikit-learn's
    mixed continuous-discrete MI). Estimates the FULL multivariate MI directly, with no clustering
    coarse-graining, so it is a tighter lower bound on I(A; representation) than a Leiden-cluster MI.

    X: (N, d) continuous features (PCA-reduce high-d embeddings first); y: (N,) discrete labels.
    """
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y)
    N = len(y)
    radius = np.zeros(N); label_counts = np.zeros(N)
    for lab in np.unique(y):
        m = y == lab; cnt = int(m.sum())
        label_counts[m] = cnt
        if cnt > 1:
            kk = min(k, cnt - 1)
            nn = NearestNeighbors(n_neighbors=kk).fit(X[m])
            d, _ = nn.kneighbors()
            radius[m] = np.nextafter(d[:, -1], 0)      # strictly inside the k-th same-class NN
    keep = label_counts > 1
    if keep.sum() < 2 or len(np.unique(y[keep])) < 2:
        return 0.0
    Xk, rk, lc = X[keep], radius[keep], label_counts[keep]
    nn_all = NearestNeighbors().fit(Xk)
    m_all = np.array([len(a) for a in
                      nn_all.radius_neighbors(Xk, radius=rk, return_distance=False)])  # includes self
    mi = (digamma(len(Xk)) + digamma(k)
          - np.mean(digamma(lc)) - np.mean(digamma(np.maximum(m_all, 1))))
    return max(mi / np.log(2.0), 0.0)


def mi_ross_report(X, y, k: int = 3, n_perm: int = 200, seed: int = 0) -> dict:
    """Ross MI in bits with a label-permutation null: excess = mi - mean(perm), p, U = excess/H(y).

    U normalises the excess, not the raw estimate, for the same reason as in mi_report: the kNN
    estimator carries its own finite-N bias, and only the excess puts chance at U = 0.
    """
    y = np.asarray(y)
    mi = mi_ross(X, y, k)
    rng = np.random.default_rng(seed)
    yy = y.copy(); perm = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(yy); perm[i] = mi_ross(X, yy, k)
    Hy = entropy_bits(y)
    excess = mi - perm.mean()
    return {"mi": mi, "excess": excess, "perm_mean": float(perm.mean()),
            "p": float((perm >= mi).mean()), "H_a": Hy,
            "U": float(excess / Hy) if Hy > 0 else 0.0}


def mi_report(a, b, n_perm: int = 1000, seed: int = 0) -> dict:
    """MI(a;b) in bits: plug-in, Miller-Madow, permutation-excess (primary) + p-value + U.

    a = ground-truth epitope labels, b = representation labels (cluster / predicted / gene).
    Excess = plug-in(a,b) - mean_perm(plug-in(a, shuffle(b))); p = P(perm >= observed).

    U = excess / H(a), NOT Miller-Madow / H(a): on a sparse table (V/J tuples, say) Miller-Madow
    leaves enough finite-N bias that randomly labelled receptors read U ~ 0.5, whereas the excess
    subtracts that bias by construction and puts chance at U = 0. Normalising the same quantity
    that is reported as bits also keeps the two columns of a report consistent.
    """
    a = np.asarray(a); b = np.asarray(b)
    M = _counts(a, b)
    plug = _mi_plugin(M)
    mm = _mi_millermadow(M)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    bb = b.copy()
    for i in range(n_perm):
        rng.shuffle(bb)
        perm[i] = _mi_plugin(_counts(a, bb))
    excess = plug - perm.mean()
    p = float((perm >= plug).mean())
    Ha = entropy_bits(a)
    return {"plugin": plug, "mm": mm, "excess": excess, "perm_mean": float(perm.mean()),
            "p": p, "H_a": Ha, "U": float(excess / Ha) if Ha > 0 else 0.0}


if __name__ == "__main__":                                    # self-check
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 600)
    b_dep = a.copy(); flip = rng.random(600) < 0.3; b_dep[flip] = rng.integers(0, 5, flip.sum())
    b_ind = rng.integers(0, 5, 600)
    dep = mi_report(a, b_dep, n_perm=500); ind = mi_report(a, b_ind, n_perm=500)
    print("dependent  :", {k: round(v, 3) for k, v in dep.items()})
    print("independent:", {k: round(v, 3) for k, v in ind.items()})
    assert dep["excess"] > 0.3 and dep["p"] < 0.01, "should detect dependence"
    assert abs(ind["excess"]) < 0.05 and ind["p"] > 0.05, "should read ~0 excess for independent"
    # Ross mixed continuous-discrete MI: separated Gaussians per class vs label-independent noise
    Xd = np.vstack([rng.normal(3 * a[i], 1.0, size=4) for i in range(600)])
    Xi = rng.normal(0, 1, size=(600, 4))
    rd = mi_ross_report(Xd, a, k=3, n_perm=100); ri = mi_ross_report(Xi, a, k=3, n_perm=100)
    print("ross dependent  :", {kk: round(v, 3) for kk, v in rd.items()})
    print("ross independent:", {kk: round(v, 3) for kk, v in ri.items()})
    assert rd["excess"] > 0.3 and rd["p"] < 0.05, "Ross should detect class separation"
    assert abs(ri["excess"]) < 0.1, "Ross should read ~0 for label-independent features"
    print("mi.py self-check OK")
