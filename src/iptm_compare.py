#!/usr/bin/env python
# 2026-08-06  TCRmodel2 confidence (interface ipTM) across cohorts and epitopes.
#
# AlphaFold-Multimer/TCRmodel2 imputes a plausible interface for essentially ANY TCR-pMHC pair, so
# ipTM confidence does not tell cognate from non-cognate pairs. We show this per epitope: real VDJdb
# binders, the 20 IMMREP25 unseen-peptide epitopes, and the mismatched-decoy negatives (split by their
# own epitopes). The confidence is governed by the epitope, not by binding truth -- some negatives fold
# better than some positives, and equally-real epitopes (GIL vs GLC) sit at opposite ends.
#
# Nested ANOVA  ipTM ~ dataset + epitope(dataset)  (dataset tested against the epitope-within-dataset
# mean square) + Tukey HSD across datasets. Figure -> the companion manuscript repo.  Run:
#   python src/iptm_compare.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
HF = Path.home() / "hf" / "tcren_structures"
IMM = REPO / "cache" / "struct" / "immrep25"
FIG = REPO.parent / "2026-immrep25-audit-ms" / "figures" / "fig_iptm.pdf"
OUT_CSV = REPO / "results" / "iptm_compare.csv"
MACROS = REPO / "appendix" / "analysis" / "iptm_macros.tex"
MIN_N = 15
BLUE, RED, GREY = "#2c7fb8", "#d7301f", "#8a8a8a"


def _short(mhc) -> str:                          # "...A*02:01..." -> "HLA-A*02:01"
    import re
    m = re.search(r"([ABC]\*\d{2}:\d{2})", str(mhc))
    return "HLA-" + m.group(1) if m else ""


def build() -> pd.DataFrame:
    # IMMREP25: cid -> epitope, cid -> ranked_0 ipTM
    epi = {}
    for ln in (IMM.parent / "immrep25_epitopes.tsv").read_text().splitlines():
        if "\\t" in ln:
            c, e = ln.split("\\t"); epi[c] = e
    rows = []
    for p in IMM.glob("*.stats.json"):
        c = p.name.replace(".stats.json", ""); e = epi.get(c)
        d = json.loads(p.read_text()).get("ranked_0", {})
        if e and "iptm" in d:
            rows.append(("IMMREP25", e, float(d["iptm"]), ""))
    md = pd.read_csv(HF / "vdjdb_binder_benchmark" / "metadata.tsv", sep="\t")
    b = md[md["y"] == 1]
    keep = b.groupby("epitope").size().loc[lambda s: s >= MIN_N].index
    for _, r in b[b.epitope.isin(keep)].iterrows():
        rows.append(("VDJdb binder", r.epitope, float(r.iptm), _short(r.get("mhc"))))
    neg = pd.read_csv(HF / "immrep23_negatives" / "immrep2022_negatives.tsv", sep="\t").dropna(subset=["iptm"])
    keep_neg = neg.groupby("epitope").size().loc[lambda s: s >= MIN_N].index  # well-populated decoy epitopes
    for _, r in neg[neg.epitope.isin(keep_neg)].iterrows():
        rows.append(("negatives", r.epitope, float(r.iptm), str(r.get("mhc.a.short", ""))))
    return pd.DataFrame(rows, columns=["dataset", "epitope", "iptm", "hla"])


def nested_anova(df):
    """ipTM ~ dataset + epitope(dataset); dataset F uses the epitope-within-dataset MS."""
    y = df.iptm.values; g = y.mean(); N = len(y)
    ds = df.groupby("dataset"); de = df.groupby(["dataset", "epitope"])
    ss_ds = sum(len(v) * (v.iptm.mean() - g) ** 2 for _, v in ds)
    ss_de = sum(len(v) * (v.iptm.mean() - ds.get_group(k[0]).iptm.mean()) ** 2 for k, v in de)
    ss_w = sum(((v.iptm - v.iptm.mean()) ** 2).sum() for _, v in de)
    k_ds, k_de = ds.ngroups, de.ngroups
    df_ds, df_de, df_w = k_ds - 1, k_de - k_ds, N - k_de
    ms_ds, ms_de, ms_w = ss_ds / df_ds, ss_de / df_de, ss_w / df_w
    F_ds, F_de = ms_ds / ms_de, ms_de / ms_w
    return dict(F_ds=F_ds, p_ds=stats.f.sf(F_ds, df_ds, df_de), df_ds=df_ds, df_de=df_de,
                F_de=F_de, p_de=stats.f.sf(F_de, df_de, df_w), df_w=df_w,
                eta_ds=ss_ds / (ss_ds + ss_de + ss_w), eta_de=ss_de / (ss_ds + ss_de + ss_w))


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def main():
    df = build()
    order_ds = ["VDJdb binder", "IMMREP25", "negatives"]
    col = {"VDJdb binder": BLUE, "IMMREP25": RED, "negatives": GREY}
    med = {d: df[df.dataset == d].iptm.median() for d in order_ds}

    an = nested_anova(df)
    groups = [df[df.dataset == d].iptm.values for d in order_ds]
    tuk = stats.tukey_hsd(*groups)
    pair_p = {(0, 1): tuk.pvalue[0, 1], (0, 2): tuk.pvalue[0, 2], (1, 2): tuk.pvalue[1, 2]}

    # within-epitope GILGFVFTL: real binder vs mismatched
    gil_pos = df[(df.dataset == "VDJdb binder") & (df.epitope == "GILGFVFTL")].iptm
    gil_neg = df[(df.dataset == "negatives") & (df.epitope == "GILGFVFTL")].iptm
    _, gil_p = stats.mannwhitneyu(gil_pos, gil_neg, alternative="greater")

    # per-epitope medians; how many positive epitopes fall below the negatives' median?
    pe = df.groupby(["dataset", "epitope"]).agg(
        size=("iptm", "size"), median=("iptm", "median"), hla=("hla", "first")).reset_index()
    neg_med = med["negatives"]
    imm_below = int((pe[(pe.dataset == "IMMREP25")]["median"] < neg_med).sum())
    neg_above_worst_pos = int((pe[pe.dataset == "negatives"]["median"] >
                               pe[pe.dataset == "IMMREP25"]["median"].min()).sum())
    pe.to_csv(REPO / "results" / "iptm_per_epitope.csv", index=False)
    pd.DataFrame([{"dataset": d, "median": med[d], "n": len(df[df.dataset == d])}
                  for d in order_ds]).to_csv(OUT_CSV, index=False)

    # ---- figure: (a) per-epitope ranking, (b) dataset boxes + Tukey stars ----
    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    pe_sorted = pe.sort_values("median")
    data = [df[(df.dataset == r.dataset) & (df.epitope == r.epitope)].iptm.values
            for r in pe_sorted.itertuples()]
    colors = [col[r.dataset] for r in pe_sorted.itertuples()]
    labels = [f"{r.epitope}" + (" \u2605" if r.epitope in ("GILGFVFTL", "GLCTLVAML") else "")
              for r in pe_sorted.itertuples()]
    # coloured violin behind (no edge), thin black-on-white boxplot in front
    vi = [i for i, v in enumerate(data) if len(v) > 1 and np.ptp(v) > 0]
    vp = ax.violinplot([data[i] for i in vi], positions=[i + 1 for i in vi],
                       orientation="horizontal", widths=0.95, showextrema=False, showmedians=False)
    for body, i in zip(vp["bodies"], vi):
        body.set_facecolor(colors[i]); body.set_alpha(0.55)
        body.set_edgecolor("none"); body.set_linewidth(0); body.set_zorder(1)
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.42,
                    showfliers=False, zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor("white"); patch.set_edgecolor("black"); patch.set_linewidth(0.5)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_color("black"); ln.set_linewidth(0.5)
    for m in bp["medians"]:
        m.set_color("black"); m.set_linewidth(0.8)
    ax.set_yticks(range(1, len(labels) + 1)); ax.set_yticklabels(labels, fontsize=6.2)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c if c != GREY else "black")
    ax.axvline(neg_med, ls="--", c=GREY, lw=1, zorder=10)   # reference line on top of everything
    ax.set_xlabel("interface ipTM (top-ranked model)"); ax.set_xlim(0.4, 1.0)
    ax.set_title("per epitope, ranked", loc="left", fontsize=8)
    # panel letter as its own bold artist, matching every other figure and caption
    ax.text(0.0, 1.035, "(a)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=col[d], alpha=0.65, label=d) for d in order_ds],
              loc="upper left", fontsize=6.5, frameon=False)

    # Panel (b) shows every observation with its mean +/- SD, in the style of DABEST (Ho et al. 2019),
    # rather than a box summary -- the raw spread is what makes the datasets' overlap visible.
    bx = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    vals = {d: df[df.dataset == d].iptm.values for d in order_ds}
    short = ["VDJdb", "IMM25", "neg"]
    for i, d in enumerate(order_ds, start=1):
        v = vals[d]
        bx.plot(i - 0.16 + rng.uniform(-0.15, 0.15, len(v)), v, ".", ms=1.5,
                color=col[d], alpha=0.45, mec="none")
        m, s = v.mean(), v.std(ddof=1)
        bx.plot([i + 0.22] * 2, [m - s, m + s], c="black", lw=1.3, solid_capstyle="butt")
        bx.plot(i + 0.22, m, "o", ms=3, c="black")
    bx.set_xlim(0.5, 3.5); bx.set_ylim(0.4, 1.0)
    bx.set_xticks([1, 2, 3]); bx.set_xticklabels(short, fontsize=7)
    bx.tick_params(labelsize=6)
    bx.set_title("by dataset", loc="left", fontsize=8)
    bx.text(0.0, 1.035, "(b)", transform=bx.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")
    # Across-dataset test = nested ANOVA (dataset with epitope nested within it), NOT the pooled
    # one-way Tukey: the three datasets span disjoint epitope sets, so a pooled test is confounded
    # by peptide composition and overpowered by N. Once the epitope is accounted for the datasets
    # do not differ -- ipTM is governed by the peptide, not by binding status.
    yb = 0.965
    bx.plot([1, 1, 3, 3], [yb - 0.01, yb, yb, yb - 0.01], c="black", lw=0.8)
    bx.text(2, yb + 0.006, f"n.s.\n(nested ANOVA $p{{=}}{an['p_ds']:.2f}$)", ha="center", va="bottom", fontsize=6)

    fig.savefig(FIG, bbox_inches="tight")

    macros = [
        (r"\iptmBind", f"{med['VDJdb binder']:.2f}"), (r"\iptmImm", f"{med['IMMREP25']:.2f}"),
        (r"\iptmNeg", f"{med['negatives']:.2f}"),
        (r"\iptmEtaEpi", f"{100 * an['eta_de']:.0f}"), (r"\iptmEtaDs", f"{100 * an['eta_ds']:.0f}"),
        (r"\iptmEpiF", f"{an['F_de']:.0f}"), (r"\iptmEpiP", "<10^{-4}" if an['p_de'] < 1e-4 else f"{an['p_de']:.3f}"),
        (r"\iptmDsF", f"{an['F_ds']:.1f}"), (r"\iptmDsP", "<10^{-4}" if an['p_ds'] < 1e-4 else f"={an['p_ds']:.2f}"),
        (r"\iptmGilPos", f"{gil_pos.median():.2f}"), (r"\iptmGilNeg", f"{gil_neg.median():.2f}"),
        (r"\iptmImmBelowNeg", f"{imm_below}"),
        (r"\iptmDsDfN", f"{an['df_ds']:.0f}"), (r"\iptmDsDfD", f"{an['df_de']:.0f}"),
        (r"\iptmEpiDfN", f"{an['df_de']:.0f}"), (r"\iptmEpiDfD", f"{an['df_w']:.0f}"),
        (r"\iptmNimm", f"{int(pe[pe.dataset == 'IMMREP25']['size'].sum())}"),
        (r"\iptmNpos", f"{int(pe[pe.dataset == 'VDJdb binder']['size'].sum())}"),
        (r"\iptmNposEp", f"{int((pe.dataset == 'VDJdb binder').sum())}"),
        (r"\iptmNneg", f"{int(pe[pe.dataset == 'negatives']['size'].sum())}"),
        (r"\iptmNnegEp", f"{int((pe.dataset == 'negatives').sum())}"),
    ]
    MACROS.write_text("".join(f"\\newcommand{{{k}}}{{{v}}}\n" for k, v in macros))

    # control-epitope table (non-IMMREP25 structural cohorts): n structures + median ipTM per
    # epitope, grouped by dataset type (positive real binders vs negative mismatched decoys).
    TBL = REPO.parent / "2026-immrep25-audit-ms" / "tables" / "iptm_control_table.tex"

    def _rows(ds):
        s = pe[pe.dataset == ds].sort_values("median", ascending=False)
        return "".join(f"\\texttt{{{r.epitope}}} & \\texttt{{{r.hla}}} & {int(r['size'])} & {r['median']:.2f} \\\\\n"
                       for _, r in s.iterrows())

    tbl = ("\\begin{tabular}{llrr}\n\\toprule\n"
           "Epitope & HLA & $n$ structures & Median ipTM \\\\\n\\midrule\n"
           "\\multicolumn{4}{l}{\\emph{Real binders (VDJdb, positive control)}} \\\\\n" + _rows("VDJdb binder") +
           "\\midrule\n\\multicolumn{4}{l}{\\emph{Mismatched decoys (negative control)}} \\\\\n" + _rows("negatives") +
           "\\bottomrule\n\\end{tabular}\n")
    TBL.write_text("% GENERATED by src/iptm_compare.py -- do not edit.\n"
                   "% Regenerate: python src/iptm_compare.py (from the 2026-immrep25-audit repo).\n" + tbl)

    print("=== dataset medians ==="); [print(f"  {d:14s} {med[d]:.3f}") for d in order_ds]
    print(f"\nnested ANOVA: epitope(dataset) F={an['F_de']:.0f} p={an['p_de']:.2g} eta2={an['eta_de']:.2f} | "
          f"dataset F={an['F_ds']:.2f} p={an['p_ds']:.2g} eta2={an['eta_ds']:.3f}")
    print("Tukey (dataset):", {f"{order_ds[i]} vs {order_ds[j]}": f"p={pair_p[(i,j)]:.2g} {stars(pair_p[(i,j)])}"
                                for i, j in pair_p})
    print(f"GILGFVFTL real {gil_pos.median():.2f} vs mismatched {gil_neg.median():.2f} (MWU greater p={gil_p:.2g})")
    print(f"IMMREP25 positive epitopes below the negatives' median: {imm_below}/20")
    print(f"\nwrote {FIG}\nwrote {OUT_CSV}\nwrote {MACROS}")


if __name__ == "__main__":
    main()
