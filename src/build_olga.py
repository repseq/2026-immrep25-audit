"""Build two OLGA control cohorts from the parallel-generated pool (cache/olga_pool_*.tsv):

  OLGA random        -- 1000 sequences sampled uniformly from the pool (raw generation)
  OLGA pgen-matched  -- 1000 sequences whose per-chain pgen matches immrep25 positives

Each is paired alpha<->beta at random into immrep-shaped synthetic epitope groups.
Run scripts/gen_olga_pool.sh first.
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts
from olga_control import compute_pgen, pgen_match

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache")
RESULTS = os.path.join(REPO, "results"); os.makedirs(RESULTS, exist_ok=True)


def load_pool(chain):
    p = os.path.join(CACHE, "olga_pool_%s.tsv" % chain)
    return pd.read_csv(p, sep="\t", names=["cdr3", "v", "j", "pgen"])


def _assemble(a, b, name, n_ep, rng):
    n = min(len(a), len(b))
    a, b = a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)
    order = rng.permutation(n)
    grp = np.arange(n) % n_ep
    return pd.DataFrame({
        "cohort": name, "tcr_id": ["%s_%d" % (name, i) for i in range(n)],
        "epitope": ["%s_ep%02d" % (name, g) for g in grp],
        "cdr3a": a.cdr3.values, "va": a.v.values, "ja": a.j.values,
        "cdr3b": b.cdr3.values[order], "vb": b.v.values[order], "jb": b.j.values[order],
    })


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    imm = build_cohorts()["immrep25_pos"]["paired"]
    n_tcr, n_ep = len(imm), imm.epitope.nunique()
    pool_a, pool_b = load_pool("A"), load_pool("B")
    t = time.time()
    pa = compute_pgen(imm.cdr3a.tolist(), "A")
    pb = compute_pgen(imm.cdr3b.tolist(), "B")
    np.savez(os.path.join(CACHE, "immrep_pgen.npz"), pa=pa, pb=pb)

    # OLGA random: uniform sample from the pool
    ra = pool_a[pool_a.pgen > 0].sample(n_tcr, random_state=1).reset_index(drop=True)
    rb = pool_b[pool_b.pgen > 0].sample(n_tcr, random_state=2).reset_index(drop=True)
    _assemble(ra, rb, "olga_random", n_ep, rng).to_csv(
        os.path.join(RESULTS, "olga_random.tsv"), sep="\t", index=False)

    # OLGA pgen-matched: stratified to immrep25 per-chain pgen
    ma = pgen_match(pa, pool_a, n_tcr, rng)
    mb = pgen_match(pb, pool_b, n_tcr, rng)
    _assemble(ma, mb, "olga_matched", n_ep, rng).to_csv(
        os.path.join(RESULTS, "olga_matched.tsv"), sep="\t", index=False)

    print("built olga_random (%d) + olga_matched (%d) in %.0fs" %
          (n_tcr, min(len(ma), len(mb)), time.time() - t))
