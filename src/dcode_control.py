"""Does an INDEPENDENT dextramer-sorted positive set carry the CDR3 convergence IMMREP25 lacks?
# 2026-08-06  (reviewer m10)

Runs the model-free homology S/N probe (identical to the main audit) on the 10x dCODE dextramer
control (src/build_dcode.py), and on its HLA-A*02:01 subset -- the same restriction as every
IMMREP25 peptide. If the dextramer set separates by epitope where IMMREP25 floors, then IMMREP25's
flatness is a property of its labels/assay, not intrinsic hardness of A*02:01 nonamers.

Usage: uv run python src/dcode_control.py   # writes results/dcode.csv + macros
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from homology import per_epitope_sn, dataset_sn                # noqa: E402
from load_data import valid_cdr3                               # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
TABLES = os.path.join(os.path.dirname(REPO), "2026-immrep25-audit-ms", "tables")
MIN_N = 30


def _long(df, chain_col, chain):
    lc = df[["epitope", chain_col]].rename(columns={chain_col: "cdr3"}).dropna()
    lc = lc[lc.cdr3.map(valid_cdr3)]
    return lc.assign(chain=chain)


def _sn(long_chain):
    pe = per_epitope_sn(long_chain, max_d=3, min_n=MIN_N, seed=1)
    return pe, (dataset_sn(pe, 1)["sn"] if len(pe) else np.nan)


def run():
    df = pd.read_csv(os.path.join(REPO, "dump", "dcode", "dcode_clonotypes.tsv"), sep="\t")
    hla = df.groupby("epitope").hla.first()
    out = {}
    for tag, sub in (("all", df), ("a02", df[df.hla == "A*02:01"])):
        peb, snb = _sn(_long(sub, "cdr3b", "B"))
        pea, sna = _sn(_long(sub, "cdr3a", "A"))
        out[tag] = dict(snb=snb, sna=sna, nep=peb.epitope.nunique() if len(peb) else 0,
                        nepa=pea.epitope.nunique() if len(pea) else 0,
                        n=len(sub.drop_duplicates(["cdr3a", "cdr3b"])), peb=peb, pea=pea)
        print("dcode-%-3s  homology S/N  beta=%.2f (%d ep)  alpha=%.2f (%d ep)  %d clonotypes"
              % (tag, snb, out[tag]["nep"], sna, out[tag]["nepa"], out[tag]["n"]))

    # per-epitope table, both chains (the probe is run on each separately), with HLA
    peb = out["all"]["peb"].copy()
    pea = out["all"]["pea"][["epitope", "n", "sn1"]].rename(columns={"n": "nA", "sn1": "snA"})
    peb = peb.merge(pea, on="epitope", how="left")
    peb["hla"] = peb.epitope.map(hla)
    peb = peb.sort_values("sn1", ascending=False)
    peb[["epitope", "hla", "n", "sn1", "nA", "snA"]].to_csv(
        os.path.join(RESULTS, "dcode.csv"), index=False)

    macros = {
        "dcodeHomB": "%.1f" % out["all"]["snb"], "dcodeHomA": "%.1f" % out["all"]["sna"],
        "dcodeHomBAtwo": "%.1f" % out["a02"]["snb"], "dcodeHomAAtwo": "%.1f" % out["a02"]["sna"],
        "dcodeNep": "%d" % out["all"]["nep"], "dcodeNepAtwo": "%d" % out["a02"]["nep"],
        "dcodeNepA": "%d" % out["all"]["nepa"], "dcodeNepAAtwo": "%d" % out["a02"]["nepa"],
        "dcodeNclono": "%d" % out["all"]["n"],
        "dcodeMaxSnb": "%.0f" % peb.sn1.max(),
        "dcodeMaxSnbEp": "\\texttt{%s}" % peb.iloc[0].epitope,
    }
    # The aggregate geometric means of dCODE and IMMREP25 are close, because both sets contain many
    # epitopes at the floor; what separates them is the ceiling. Report how many A*02:01 dextramer
    # epitopes beat IMMREP25's single best peptide, and their S/N values.
    IMMREP25_MAX_SNB = 25.05        # \pepMaxSnb, run_audit.py per-epitope table (YLFNADIWI)
    a02 = peb[peb.hla == "A*02:01"].sort_values("sn1", ascending=False)
    top = a02[a02.sn1 > IMMREP25_MAX_SNB]
    def _prose_list(items):
        """a, b and c -- so the macro drops straight into a sentence."""
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    # the per-epitope contrast Note 3 quotes: the strongest alpha-vs-beta split, and the epitope
    # whose alpha carries signal while its beta does not
    for ep in ("FLYALALLL", "LLDFVRFMGV"):
        r = peb[peb.epitope == ep]           # peb already carries the alpha column as snA
        macros["dcodeSnb%s" % ep] = ("%.0f" % r.sn1.iloc[0]) if len(r) else "--"
        macros["dcodeSna%s" % ep] = ("%.0f" % r.snA.iloc[0]) if len(r) else "--"
    macros["dcodeNepAtwoAboveImm"] = "%d" % len(top)
    macros["dcodeSnbAtwoAboveImm"] = _prose_list("%.0f" % v for v in top.sn1)
    macros["dcodeEpAtwoAboveImm"] = _prose_list("\\texttt{%s}" % e for e in top.epitope)
    print("A*02:01 dextramer epitopes above IMMREP25's best (%.1f): %s"
          % (IMMREP25_MAX_SNB, ", ".join("%s=%.1f" % (e, v)
                                         for e, v in zip(top.epitope, top.sn1)) or "none"))
    with open(os.path.join(ADAT, "dcode_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    with open(os.path.join(TABLES, "dcode_table.tex"), "w") as fh:
        fh.write("%% GENERATED by src/dcode_control.py -- do not edit.\n"
                 "%% Regenerate: python src/dcode_control.py (from the 2026-immrep25-audit repo).\n")
        fh.write("\\begin{tabular}{llrrrr}\n\\toprule\n"
                 " & & \\multicolumn{2}{c}{TCR$\\beta$} & \\multicolumn{2}{c}{TCR$\\alpha$} \\\\\n"
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
                 "epitope & HLA & clonotypes & S/N & clonotypes & S/N \\\\\n\\midrule\n")
        for _, r in peb.iterrows():
            na = "--" if pd.isna(r.nA) else "%d" % int(r.nA)
            sa = "--" if pd.isna(r.snA) else "%.1f" % r.snA
            fh.write("\\texttt{%s} & %s & %d & %.1f & %s & %s \\\\\n"
                     % (r.epitope, r.hla, int(r.n), r.sn1, na, sa))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nA*02:01 only: dcode beta S/N %.1f vs IMMREP25 A*02:01 (MIRA) at ~floor -- same HLA, "
          "different assay." % out["a02"]["snb"])
    print("wrote results/dcode.csv + dcode_macros.tex")


if __name__ == "__main__":
    run()
