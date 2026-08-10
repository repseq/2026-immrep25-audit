"""One global PCA of the ESM-2 embedding space, fit on every unique clonotype in the study.
# 2026-08-09

The kNN mutual-information estimator (mi.mi_ross) needs a low-dimensional representation: kNN
distances concentrate in 480 dimensions, so the estimate is unreliable on the raw embedding. The
projection used for that was previously fit inside each cohort, which makes the axes cohort-specific
and the resulting bits not strictly comparable across cohorts.

Here the projection is fit ONCE per chain, on the union of unique CDR3s across all cohorts, and
every cohort is then projected onto the same axes. This is legitimate -- and preferable -- because
PCA is unsupervised: it never sees an epitope label, so it cannot leak label information into an
estimate of I(epitope; representation). The quantity being reported is how much epitope information a
fixed representation carries, not the generalisation of a shipped classifier; the only step that must
stay inside cross-validation folds is the SUPERVISED fit of the held-out probe (degenerate.py), which
is untouched.

Usage: python src/esm_pca.py     # writes cache/esm/pca_{A,B}.npz  (+ allseq_{A,B}.npy)
"""
from __future__ import annotations
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from cohorts import build_cohorts, HIERARCHY      # noqa: E402
from esm_signal import CACHE, PCA_DIM, embed      # noqa: E402


def _pca_path(chain: str) -> str:
    return os.path.join(CACHE, "pca_%s.npz" % chain)


def project(X: np.ndarray, chain: str) -> np.ndarray:
    """Project cohort embeddings onto the global axes for `chain` (build_pca must have run)."""
    p = np.load(_pca_path(chain))
    return (X - p["mean"]) @ p["components"].T


def build_pca():
    coh = build_cohorts(include_olga=True)
    for chain in ("A", "B"):
        seqs = sorted({s for name in HIERARCHY if name in coh
                       for s in coh[name]["long"].query("chain == @chain").cdr3.dropna()})
        cache = os.path.join(CACHE, "allseq_%s.npy" % chain)
        if os.path.exists(cache) and np.load(cache, mmap_mode="r").shape[0] == len(seqs):
            X = np.load(cache)
        else:
            X = embed(seqs)
            np.save(cache, X)
        from sklearn.decomposition import PCA
        p = PCA(n_components=PCA_DIM, random_state=0).fit(X)
        np.savez(_pca_path(chain), mean=p.mean_, components=p.components_,
                 evr=p.explained_variance_ratio_, n=len(seqs))
        print("TR%s  %d unique CDR3 -> %d components, %.1f%% of the embedding variance"
              % (chain, len(seqs), PCA_DIM, 100 * p.explained_variance_ratio_.sum()))
    print("\nwrote cache/esm/pca_{A,B}.npz")


if __name__ == "__main__":
    build_pca()
