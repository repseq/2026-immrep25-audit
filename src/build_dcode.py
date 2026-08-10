"""Build the 10x dCODE-Dextramer independent positive control (reviewer m10).
# 2026-08-06

10x Genomics dCODE Dextramer dataset (4 donors, "A New Way of Exploring Immunity", 2019): single
cells carry CITE-seq surface markers, paired TCR clonotypes, and binarized pMHC-dextramer binding
calls (`*_binder` = True/False) against ~44 real dextramers plus negative-control dextramers.
Dextramer sorting is an INDEPENDENT validation method from the MIRA assay behind IMMREP25, so this
is the "independently validated unseen-style positive control" the review asked for.

We keep cells with exactly one real-dextramer binder (unambiguous specificity; negative-control
dextramers excluded), read the epitope + HLA from the dextramer name and the CDR3s from the
clonotype string, and deduplicate to unique clonotypes per epitope. Output mirrors the other
cohorts' paired schema so the homology probe runs unchanged.

Usage: uv run python src/build_dcode.py   # writes dump/dcode/dcode_clonotypes.tsv
"""
from __future__ import annotations
import os, csv, gzip
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.expanduser("~/hf/airr_benchmark/dcode")
OUT = os.path.join(REPO, "dump", "dcode")


def _cdr3(clono: str, chain: str) -> str | None:
    """First CDR3 of the requested chain from a '10x TRA:..;TRB:..' clonotype string."""
    for part in clono.split(";"):
        if part.startswith(chain + ":"):
            return part.split(":", 1)[1]
    return None


def _is_control(binder_col: str) -> bool:
    name = binder_col[:-7]                       # strip '_binder'
    f = name.split("_")
    return name.startswith("NR(") or "NC" in f   # negative-control dextramers


def run():
    rows = []
    for d in (1, 2, 3, 4):
        path = os.path.join(SRC, f"vdj_v1_hs_aggregated_donor{d}_binarized_matrix.csv.gz")
        with gzip.open(path, "rt") as fh:
            r = csv.reader(fh); h = next(r)
            bcols = {i: h[i] for i in range(len(h))
                     if h[i].endswith("_binder") and not _is_control(h[i])}
            ci = h.index("cell_clono_cdr3_aa")
            for row in r:
                hits = [bcols[i] for i in bcols if row[i] == "True"]
                if len(hits) != 1:
                    continue                     # require unambiguous single specificity
                dextr = hits[0][:-7].split("_")  # e.g. A0201_GILGFVFTL_Flu-MP_Influenza
                hla, pep = dextr[0], dextr[1]
                hla = hla[0] + "*" + hla[1:3] + ":" + hla[3:5]        # A0201 -> A*02:01
                cb, ca = _cdr3(row[ci], "TRB"), _cdr3(row[ci], "TRA")
                if cb:
                    rows.append({"donor": d, "epitope": pep, "hla": hla,
                                 "cdr3a": ca, "cdr3b": cb})
    df = pd.DataFrame(rows).drop_duplicates(["epitope", "cdr3a", "cdr3b"]).reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "dcode_clonotypes.tsv"), sep="\t", index=False)
    vc = df.epitope.value_counts()
    print("unique dextramer clonotypes:", len(df), "| epitopes >=30:", int((vc >= 30).sum()))
    print(vc[vc >= 30].head(15).to_string())
    print("wrote dump/dcode/dcode_clonotypes.tsv")


if __name__ == "__main__":
    run()
