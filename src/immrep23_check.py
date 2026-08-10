"""Robustness check on the S/N ladder: the seen-peptide benchmark at 32 epitopes, not 7.
# 2026-08-10

IMMREP22 is the ladder's seen-peptide rung, and it carries only 7 qualifying epitopes, so its
homology S/N rests on a narrow and unusually deeply-studied set (GILGFVFTL, NLVPMVATV, YLQPRTFLL and
four others). This asks whether the rung survives a much broader seen benchmark: IMMREP23's training
positives (Nielsen 2024) pooled with IMMREP22's -- 11241 paired clonotypes over 808 peptides, 32 of
them qualifying, four and a half times IMMREP22's epitope count.

It is a CHECK, not a cohort. The pooled set is VDJdb-derived and so is not independent of the VDJdb
rungs above it; and on the learned probes (embedding clustering, held-out transfer) it behaves quite
differently from IMMREP22, because those probes are sensitive to how deeply an epitope has been
studied in a way the one-vs-many homology statistic is not. Only the model-free homology S/N is
reported here, and only to show that the ladder's ordering does not depend on IMMREP22's seven
epitopes.

Usage: python src/immrep23_check.py   # writes appendix/analysis/immrep23_macros.tex
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from load_data import load_immrep23, load_immrep22, valid_cdr3, IMMREP23_TRAIN   # noqa: E402
from cohorts import _mk_paired, _pair_to_long            # noqa: E402
from homology import per_epitope_sn, dataset_sn          # noqa: E402

ADAT = os.path.join(REPO, "appendix", "analysis")
MIN_N = 30


def run():
    long = _pair_to_long(_mk_paired(load_immrep23(), "immrep23", "true"))
    m = {}
    for chain in ("A", "B"):
        lc = long[long.chain == chain][["epitope", "cdr3"]].dropna()
        lc = lc[lc.cdr3.map(valid_cdr3)].drop_duplicates(["epitope", "cdr3"]).reset_index(drop=True)
        pe = per_epitope_sn(lc, max_d=1, min_n=MIN_N)
        ds = dataset_sn(pe, d=1)
        m["chkHom%s" % chain] = "%.2f" % ds["sn"]
        m["chkLo%s" % chain] = "%.2f" % ds["lo"]
        m["chkHi%s" % chain] = "%.2f" % ds["hi"]
        m["chkNep%s" % chain] = "%d" % ds["n_ep"]
        print("TR%s: S/N %.2f (95%% CI %.2f-%.2f) over %d qualifying epitopes, %d clonotypes"
              % (chain, ds["sn"], ds["lo"], ds["hi"], ds["n_ep"], len(lc)))
    m["chkN"] = "{:,}".format(len(long[long.chain == "B"]))
    # how far IMMREP22's positives are already inside IMMREP23's training set: the two are
    # curations of VDJdb at different times, which is why the pooled set adds no rung of its own
    i22 = load_immrep22()
    i22 = i22[i22.cdr3a.map(valid_cdr3) & i22.cdr3b.map(valid_cdr3)]
    T22 = set(zip(i22.epitope.astype(str), i22.cdr3a.astype(str), i22.cdr3b.astype(str)))
    tr = pd.read_csv(IMMREP23_TRAIN)
    T23 = set(zip(tr.Peptide.astype(str), tr.CDR3a_extended.astype(str), tr.CDR3b_extended.astype(str)))
    m["chkSeenN"] = "%d" % len(T22)
    m["chkSeenIn"] = "%d" % len(T22 & T23)
    # macro names must be letters only -- \i23HomB parses as \i followed by "23HomB"
    with open(os.path.join(ADAT, "immrep23_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote appendix/analysis/immrep23_macros.tex")


if __name__ == "__main__":
    run()
