"""1-mm neighbourhood pgen (as in mirpy): P(all CDR3s within Hamming distance 1).

Uses OLGA's compute_hamming_dist_1_pgen (wildcard trick: L+1 model calls). This is a
generation-degree measure -- how dense a sequence's neighbourhood is in recombination
space -- and controls homology matchability better than the point pgen. Terminal
anchor positions are skipped (they are germline-encoded), matching mirpy's default.

Writes cache/pgen1mm_<cohort>_<chain>.tsv with columns: cdr3 pgen pgen1mm.
"""
from __future__ import annotations
import os, sys, multiprocessing as mp
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache"); os.makedirs(CACHE, exist_ok=True)
SKIP_ENDS = 2

_M = None


def _init(chain):
    global _M
    sys.path.insert(0, os.path.join(REPO, "src"))
    from olga_control import load_models
    _M, _, _ = load_models(chain)


def _work(seq):
    try:
        p0 = float(_M.compute_aa_CDR3_pgen(seq))
        p1 = float(_M.compute_hamming_dist_1_pgen(seq, print_warnings=False))
    except Exception:
        p0, p1 = 0.0, 0.0
    return (seq, p0, p1)


def compute(seqs, chain, nproc=8):
    seqs = [s for s in dict.fromkeys(seqs) if isinstance(s, str) and len(s) > SKIP_ENDS * 2 + 1]
    with mp.Pool(nproc, initializer=_init, initargs=(chain,)) as pool:
        rows = pool.map(_work, seqs, chunksize=16)
    return pd.DataFrame(rows, columns=["cdr3", "pgen", "pgen1mm"])


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "src"))
    from cohorts import build_cohorts
    import time
    coh = build_cohorts(include_olga=True)
    cohorts = ["immrep25_pos", "olga_matched", "olga_random", "airr_control"]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for name in cohorts:
        for chain in ("A", "B"):
            col = "cdr3a" if chain == "A" else "cdr3b"
            seqs = coh[name]["paired"][col].tolist()
            t = time.time()
            df = compute(seqs, chain, nproc)
            df.to_csv(os.path.join(CACHE, "pgen1mm_%s_%s.tsv" % (name, chain)), sep="\t", index=False)
            print("%-16s TR%s: n=%d  %.0fs  med log10 pgen=%.2f  pgen1mm=%.2f" % (
                name, chain, len(df), time.time() - t,
                np.median(np.log10(df.pgen[df.pgen > 0])),
                np.median(np.log10(df.pgen1mm[df.pgen1mm > 0]))))
