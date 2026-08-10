#!/usr/bin/env python3
"""Regenerate src/epitope_hla.py -- the epitope -> restricting-HLA map used to annotate the
per-epitope tables (run_audit.py) and figures. Values are READ FROM DATA, never memory:
  IMMREP25            dump/immrep25/immrep2025_for_release.tsv           column `hla`
  IMMREP22 / controls ~/hf/tcren_structures/immrep23_negatives/immrep2022_negatives.tsv  `mhc.a.short`
                      ~/hf/tcren_structures/vdjdb_binder_benchmark/metadata.tsv           `mhc`
Run:  python tools/build_epitope_hla.py   (rewrites src/epitope_hla.py)
# 2026-08-07
"""
import re, os, pathlib
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
HF = pathlib.Path.home() / "hf" / "tcren_structures"


def _short(m):
    m = str(m); mm = re.search(r"([ABC]\*\d{2}:\d{2})", m)
    return "HLA-" + mm.group(1) if mm else None


def build():
    hla = {}
    # IMMREP25: hla column already "A*02:01" style -> normalise to HLA-A*02:01
    im = pd.read_csv(REPO / "dump" / "immrep25" / "immrep2025_for_release.tsv", sep="\t")
    for e, h in im.groupby("peptide")["hla"].first().items():
        hla[e] = _short(h) or ("HLA-" + str(h))
    # IMMREP22 / seen comparator + structural controls (mhc.a.short already "HLA-..-")
    neg = pd.read_csv(HF / "immrep23_negatives" / "immrep2022_negatives.tsv", sep="\t")
    for e, h in neg.groupby("epitope")["mhc.a.short"].first().items():
        hla.setdefault(e, str(h))
    vb = pd.read_csv(HF / "vdjdb_binder_benchmark" / "metadata.tsv", sep="\t")
    for e, h in vb.groupby("epitope")["mhc"].first().items():
        s = _short(h)
        if s:
            hla.setdefault(e, s)
    return dict(sorted(hla.items()))


def main():
    hla = build()
    out = REPO / "src" / "epitope_hla.py"
    lines = ['"""Epitope -> restricting HLA allele (generated; see tools/build_epitope_hla.py).',
             'Provenance: IMMREP25 from the benchmark `hla` column; IMMREP22/controls from the',
             'immrep_2022 and VDJdb-binder structure metadata (mhc). Do not hand-edit -- rerun the tool."""',
             "EPITOPE_HLA = {"]
    for e, h in hla.items():
        lines.append(f'    "{e}": "{h}",')
    lines.append("}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(hla)} epitopes)")


if __name__ == "__main__":
    main()
