"""Analyze 1-mm neighbourhood pgen: does immrep25 exceed its point-pgen-matched OLGA
control (i.e. is its homology a neighbourhood-density artefact), and does real data
(AIRR) show more generation degree than random OLGA?

Reads cache/pgen1mm_<cohort>_<chain>.tsv (from compute_1mm.py); writes a summary CSV,
a gnuplot .dat, and a matplotlib figure.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
FIGS = os.path.join(RESULTS, "figures")

COHORTS = ["immrep25_pos", "airr_control", "airr_top", "olga_matched", "olga_random"]
LAB = {"immrep25_pos": "immrep25", "airr_control": "AIRR random", "airr_top": "AIRR rank-ladder",
       "olga_matched": "OLGA pgen-ladder", "olga_random": "OLGA random"}
COL = {"immrep25_pos": "#ef8a00", "airr_control": "#9e9ac8", "airr_top": "#6a51a3",
       "olga_matched": "#b2182b", "olga_random": "#777777"}


def load(name, chain):
    p = os.path.join(CACHE, "pgen1mm_%s_%s.tsv" % (name, chain))
    return pd.read_csv(p, sep="\t") if os.path.exists(p) else None


if __name__ == "__main__":
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, chain in zip(axes, ("A", "B")):
        for name in COHORTS:
            df = load(name, chain)
            if df is None:
                continue
            l1 = np.log10(df.pgen1mm[df.pgen1mm > 0])
            l0 = np.log10(df.pgen[df.pgen > 0])
            rows.append({"cohort": name, "chain": chain, "n": len(df),
                         "med_log_pgen": float(np.median(l0)),
                         "med_log_pgen1mm": float(np.median(l1)),
                         "q25_1mm": float(np.quantile(l1, .25)),
                         "q75_1mm": float(np.quantile(l1, .75))})
            ax.hist(l1, bins=40, density=True, histtype="step", lw=2,
                    color=COL[name], label=LAB[name])
        ax.set_title("TCR%s" % ("$\\alpha$" if chain == "A" else "$\\beta$"))
        ax.set_xlabel("log10 1-mm neighbourhood pgen"); ax.set_ylabel("density")
        if chain == "A":
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_pgen1mm.pdf")); plt.close(fig)

    # gnuplot .dat: one column of log10 1-mm pgen per cohort/chain
    for name in COHORTS:
        for chain in ("A", "B"):
            df = load(name, chain)
            if df is None:
                continue
            v = np.log10(df.pgen1mm[df.pgen1mm > 0])
            with open(os.path.join(ADAT, "pgen1mm_%s_%s.dat" % (name, chain)), "w") as fh:
                fh.write("# log10pgen1mm\n")
                v.to_csv(fh, index=False, header=False)

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS, "pgen1mm_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print("\n[key contrasts, median log10 1-mm pgen]")
    for chain in ("A", "B"):
        s = summ[summ.chain == chain].set_index("cohort")["med_log_pgen1mm"]
        print("  TR%s: immrep=%.2f OLGA-m=%.2f (imm-m=%+.2f) | AIRR-uniq=%.2f AIRR-top=%.2f OLGA-r=%.2f"
              % (chain, s.get("immrep25_pos", np.nan), s.get("olga_matched", np.nan),
                 s.get("immrep25_pos", np.nan) - s.get("olga_matched", np.nan),
                 s.get("airr_control", np.nan), s.get("airr_top", np.nan), s.get("olga_random", np.nan)))
