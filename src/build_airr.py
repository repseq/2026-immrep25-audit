"""Build a real post-thymic-selection repertoire control from isalgo/airr_control.

Reservoir-samples 1000 unique TRA and 1000 unique TRB clonotypes (vdjtools .aa
format: cdr3aa, v, j) from the pooled human control, pairs alpha<->beta at random
into immrep-shaped synthetic epitope groups. Real sequences (post-selection) with no
epitope structure -> a real-data analogue of the OLGA floor.

Files are large and gitignored; fetch with scripts/fetch_airr.sh first.
"""
import os, sys, gzip, random
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import valid_cdr3, canon_gene
from cohorts import build_cohorts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIRR = os.path.join(REPO, "cache", "airr")
RESULTS = os.path.join(REPO, "results")


def reservoir(path, n, seed):
    """Uniformly sample n valid (cdr3, v, j) rows from a gzipped vdjtools .aa stream."""
    rng = random.Random(seed)
    res, k = [], 0
    with gzip.open(path, "rt") as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            cdr3, v, j = p[3], canon_gene(p[4]), canon_gene(p[6])
            if not valid_cdr3(cdr3):
                continue
            if len(res) < n:
                res.append((cdr3, v, j))
            else:
                r = rng.randint(0, k)
                if r < n:
                    res[r] = (cdr3, v, j)
            k += 1
    return pd.DataFrame(res, columns=["cdr3", "v", "j"])


def top_by_frequency(path, n):
    """First n valid (cdr3, v, j) rows -- vdjtools .aa is sorted by count descending."""
    res = []
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7 or not valid_cdr3(p[3]):
                continue
            res.append((p[3], canon_gene(p[4]), canon_gene(p[6])))
            if len(res) >= n:
                break
    return pd.DataFrame(res, columns=["cdr3", "v", "j"])


def _assemble(a, b, name, epis, rng):
    n = min(len(a), len(b)); a, b = a.iloc[:n], b.iloc[:n]
    order = rng.permutation(n)                       # random alpha<->beta pairing
    return pd.DataFrame({
        "cohort": name, "tcr_id": ["%s_%d" % (name, i) for i in range(n)],
        "epitope": epis[:n],
        "cdr3a": a.cdr3.values, "va": a.v.values, "ja": a.j.values,
        "cdr3b": b.cdr3.values[order], "vb": b.v.values[order], "jb": b.j.values[order],
    })


if __name__ == "__main__":
    import numpy as np
    imm = build_cohorts()["immrep25_pos"]["paired"]
    n_tcr, n_ep = len(imm), imm.epitope.nunique()
    per = n_tcr // n_ep
    fa = os.path.join(AIRR, "human.tra.aa.tsv.gz"); fb = os.path.join(AIRR, "human.trb.aa.tsv.gz")
    # AIRR unique: uniform over unique clonotypes, RANDOM epitope labels (pure floor)
    ua, ub = reservoir(fa, n_tcr, 42), reservoir(fb, n_tcr, 43)
    rand_ep = ["airr_ep%02d" % (i % n_ep) for i in range(n_tcr)]
    _assemble(ua, ub, "airr_control", rand_ep, np.random.default_rng(42)).to_csv(
        os.path.join(RESULTS, "airr_control.tsv"), sep="\t", index=False)
    # AIRR rank-ladder: top-N by clone frequency, epitope = consecutive rank bins of 50
    ta, tb = top_by_frequency(fa, n_tcr), top_by_frequency(fb, n_tcr)
    ladder_ep = ["rank%02d" % (i // per) for i in range(n_tcr)]
    _assemble(ta, tb, "airr_top", ladder_ep, np.random.default_rng(44)).to_csv(
        os.path.join(RESULTS, "airr_top.tsv"), sep="\t", index=False)
    print("built airr_control (unique/random-label) + airr_top (rank-ladder, %d bins of %d)"
          % (n_ep, per))
