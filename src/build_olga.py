"""Build two OLGA control cohorts from the parallel-generated pool (cache/olga_pool_*.tsv):

  OLGA random         -- 1000 uniform samples, RANDOM synthetic epitope labels (pure floor)
  OLGA pgen-matched   -- for each of the 20 immrep25 epitopes, 50 sequences whose per-chain
                         pgen matches THAT epitope's positives; labelled by the epitope.
                         (Per-epitope, not whole-immrep -- so any pgen-driven within-epitope
                         homology would be reproduced here.)

Run scripts/gen_olga_pool.sh first.
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts
from olga_control import compute_pgen

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache")
RESULTS = os.path.join(REPO, "results"); os.makedirs(RESULTS, exist_ok=True)


def load_pool(chain):
    p = os.path.join(CACHE, "olga_pool_%s.tsv" % chain)
    df = pd.read_csv(p, sep="\t", names=["cdr3", "v", "j", "pgen"])
    df = df[df.pgen > 0].reset_index(drop=True)
    df["lp"] = np.log10(df.pgen.values)
    return df.sort_values("lp").reset_index(drop=True)


def match_pgen(target_pgen, pool, rng, k=25):
    """For each target pgen pick a random pool row among its k nearest-pgen neighbours."""
    lt = np.log10(np.maximum(np.asarray(target_pgen), 1e-30))
    lp = pool.lp.values
    idxs = []
    for x in lt:
        pos = int(np.searchsorted(lp, x))
        lo = max(0, min(pos - k // 2, len(lp) - k))
        idxs.append(lo + int(rng.integers(min(k, len(lp)))))
    return pool.iloc[idxs].reset_index(drop=True)


def _pair(a, b, cohort, epis, rng):
    n = len(a)
    return pd.DataFrame({
        "cohort": cohort, "tcr_id": ["%s_%d" % (cohort, i) for i in range(n)],
        "epitope": epis,
        "cdr3a": a.cdr3.values, "va": a.v.values, "ja": a.j.values,
        "cdr3b": b.cdr3.values, "vb": b.v.values, "jb": b.j.values,
    })


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    imm = build_cohorts()["immrep25_pos"]["paired"].reset_index(drop=True)
    pool_a, pool_b = load_pool("A"), load_pool("B")
    t = time.time()
    pa = compute_pgen(imm.cdr3a.tolist(), "A")
    pb = compute_pgen(imm.cdr3b.tolist(), "B")
    np.savez(os.path.join(CACHE, "immrep_pgen.npz"), pa=pa, pb=pb)

    # --- OLGA random: uniform sample, random epitope labels ---
    n = len(imm); n_ep = imm.epitope.nunique()
    ra = pool_a.sample(n, random_state=1).reset_index(drop=True)
    rb = pool_b.sample(n, random_state=2).reset_index(drop=True)
    _pair(ra, rb, "olga_random", ["olga_ep%02d" % (i % n_ep) for i in range(n)], rng)\
        .to_csv(os.path.join(RESULTS, "olga_random.tsv"), sep="\t", index=False)

    # --- OLGA pgen-matched PER EPITOPE ---
    parts = []
    for e, g in imm.groupby("epitope"):
        idx = g.index.values
        ma = match_pgen(pa[idx], pool_a, rng)
        mb = match_pgen(pb[idx], pool_b, rng)
        mb = mb.iloc[rng.permutation(len(mb))].reset_index(drop=True)   # random alpha<->beta pairing
        parts.append(_pair(ma, mb, "olga_matched", [e] * len(g), rng))
    pd.concat(parts, ignore_index=True).assign(
        tcr_id=lambda d: ["olga_matched_%d" % i for i in range(len(d))]
    ).to_csv(os.path.join(RESULTS, "olga_matched.tsv"), sep="\t", index=False)
    print("built olga_random (%d) + olga_matched per-epitope (%d ep x 50), %.0fs"
          % (n, n_ep, time.time() - t))
