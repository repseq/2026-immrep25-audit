"""One-off builder: construct the pgen-matched OLGA control and cache it."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts
from olga_control import build_olga_control

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache"); os.makedirs(CACHE, exist_ok=True)
RESULTS = os.path.join(REPO, "results"); os.makedirs(RESULTS, exist_ok=True)

if __name__ == "__main__":
    n_pool = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    imm = build_cohorts()["immrep25_pos"]["paired"]
    t = time.time()
    olga, extra = build_olga_control(imm, n_pool=n_pool, seed=0)
    olga.to_csv(os.path.join(RESULTS, "olga_control.tsv"), sep="\t", index=False)
    np.savez(os.path.join(CACHE, "immrep_pgen.npz"), pa=extra["pa"], pb=extra["pb"])
    extra["pool_a"].to_csv(os.path.join(CACHE, "olga_pool_a.tsv"), sep="\t", index=False)
    extra["pool_b"].to_csv(os.path.join(CACHE, "olga_pool_b.tsv"), sep="\t", index=False)
    print("OLGA control built: %d pairs, %.0fs" % (len(olga), time.time() - t))
    print(olga.head(3).to_string())
