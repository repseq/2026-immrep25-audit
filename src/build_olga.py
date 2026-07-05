"""Build the OLGA random control (raw generative floor) from the parallel-generated
pool (cache/olga_pool_*.tsv): 1000 uniform samples with random synthetic epitope labels.

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
    return df[df.pgen > 0].reset_index(drop=True)


def _pair(a, b, cohort, epis):
    n = len(a)
    return pd.DataFrame({
        "cohort": cohort, "tcr_id": ["%s_%d" % (cohort, i) for i in range(n)],
        "epitope": epis,
        "cdr3a": a.cdr3.values, "va": a.v.values, "ja": a.j.values,
        "cdr3b": b.cdr3.values, "vb": b.v.values, "jb": b.j.values,
    })


if __name__ == "__main__":
    t = time.time()
    imm = build_cohorts()["immrep25_pos"]["paired"].reset_index(drop=True)
    n, n_ep = len(imm), imm.epitope.nunique()
    pool_a, pool_b = load_pool("A"), load_pool("B")
    # cache immrep25 pgen (used by the pgen / degree figures)
    np.savez(os.path.join(CACHE, "immrep_pgen.npz"),
             pa=compute_pgen(imm.cdr3a.tolist(), "A"), pb=compute_pgen(imm.cdr3b.tolist(), "B"))
    ra = pool_a.sample(n, random_state=1).reset_index(drop=True)
    rb = pool_b.sample(n, random_state=2).reset_index(drop=True)
    _pair(ra, rb, "olga_random", ["olga_ep%02d" % (i % n_ep) for i in range(n)])\
        .to_csv(os.path.join(RESULTS, "olga_random.tsv"), sep="\t", index=False)
    print("built olga_random (%d), %.0fs" % (n, time.time() - t))
