"""Provenance of the two sources withheld from the VDJdb cohorts, and the numbers Methods quotes.
# 2026-08-09

`load_vdjdb` fixes the HQ/LQ tier from the number of distinct studies per epitope over the COMPLETE
table, then withholds two sources from the cohorts themselves (see load_data.VDJDB_EXCLUDE_*):

  * a phage-display library, contributing many TCRb variants against a single TCRa -- a mutagenesis
    series around one degenerate receptor, not a repertoire;
  * the 10x dCODE dextramer application note, which this study analyses separately as an independent
    positive control, and which therefore cannot also sit inside the cohorts it validates.

This script quantifies both -- the phage library's share of the pre-exclusion VDJdb(HQ) TCRb records,
and the clonotype overlap between the note's VDJdb records and the dCODE control -- and emits the
macros Methods uses. It changes no cohort; it only measures what the exclusions remove.

Usage: python src/vdjdb_exclusions.py   # writes appendix/analysis/vdjdb_excl_macros.tex
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from load_data import VDJDB, valid_cdr3, VDJDB_EXCLUDE_PHAGE, VDJDB_EXCLUDE_10X  # noqa: E402

DCODE = os.path.join(REPO, "dump", "dcode", "dcode_clonotypes.tsv")
MACROS = os.path.join(REPO, "appendix", "analysis", "vdjdb_excl_macros.tex")


def run():
    v = pd.read_csv(VDJDB, sep="\t", low_memory=False)
    v = v[(v["species"] == "HomoSapiens") & (v["mhc.class"] == "MHCI")].copy()
    v["epitope"] = v["antigen.epitope"].astype(str)
    v["chain"] = v["gene"].map({"TRA": "A", "TRB": "B"})
    v = v[v["cdr3"].map(valid_cdr3)]
    # tier as load_vdjdb fixes it: distinct studies per epitope, over the complete table
    epi_refs = v.groupby("epitope")["reference.id"].nunique()
    v["quality"] = np.where(v.epitope.map(epi_refs) >= 2, "hq", "lq")
    ref = v["reference.id"].astype(str)

    # -- phage library: its share of the pre-exclusion HQ beta clonotypes -----------------------
    ph = ref == VDJDB_EXCLUDE_PHAGE
    hqb = v[(v.quality == "hq") & (v.chain == "B")].drop_duplicates(["epitope", "cdr3"])
    hqb_ph = v[ph & (v.quality == "hq") & (v.chain == "B")].drop_duplicates(["epitope", "cdr3"])
    phage_pct = 100.0 * len(hqb_ph) / len(hqb)
    n_beta = int((ph & (v.chain == "B")).sum())
    n_alpha_cdr3 = int(v[ph & (v.chain == "A")].cdr3.nunique())

    # -- 10x note: clonotype overlap with the independent dCODE control -------------------------
    tx = ref.str.contains(VDJDB_EXCLUDE_10X, regex=False)
    tb = v[tx & (v.chain == "B")]
    vk = set(zip(tb.epitope, tb.cdr3))
    d = pd.read_csv(DCODE, sep="\t")
    dk = set(zip(d.epitope.astype(str), d.cdr3b.astype(str)))
    overlap_pct = 100.0 * len(vk & dk) / max(len(vk), 1)
    tx_eps = sorted(v[tx].epitope.unique())
    only_tx = [e for e in tx_eps
               if set(v[v.epitope == e]["reference.id"].astype(str)) == set(ref[tx].unique())]
    q = v.drop_duplicates("epitope").set_index("epitope")["quality"]
    only_tx_lq = sum(q[e] == "lq" for e in only_tx)

    print(f"phage library ({VDJDB_EXCLUDE_PHAGE}): {n_beta} TCRb records against "
          f"{n_alpha_cdr3} distinct TCRa CDR3; {phage_pct:.1f}% of pre-exclusion HQ TCRb clonotypes")
    print(f"10x note: {len(vk)} VDJdb TCRb clonotypes over {len(tx_eps)} epitopes; "
          f"{overlap_pct:.1f}% also in the dCODE control; "
          f"{len(only_tx)} epitopes have it as their only source ({only_tx_lq} of them LQ)")

    with open(MACROS, "w") as fh:
        fh.write("\\newcommand{\\qPhagePct}{%.0f}\n" % phage_pct)
        fh.write("\\newcommand{\\qPhageNbeta}{%d}\n" % n_beta)
        fh.write("\\newcommand{\\qPhageNalpha}{%d}\n" % n_alpha_cdr3)
        fh.write("\\newcommand{\\qTenxOverlap}{%.1f}\n" % overlap_pct)
        fh.write("\\newcommand{\\qTenxNep}{%d}\n" % len(tx_eps))
        fh.write("\\newcommand{\\qTenxOnlyNep}{%d}\n" % len(only_tx))
    print("\nwrote appendix/analysis/vdjdb_excl_macros.tex")


if __name__ == "__main__":
    run()
