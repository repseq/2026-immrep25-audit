#!/usr/bin/env python
# 2026-09-13  pLDDT distributions per peptide: IMMREP25's positives against every other modelled
#             TCR-pMHC set we hold that carries a generator confidence.
#
# Why pLDDT and not ipTM: the IMMREP25 winner's score WAS a pLDDT statistic -- Table 2 of the
# challenge preprint gives Bradley/TCRdock-AF3 at macro-AUC_0.1 0.601 using "pLDDT averaged
# across peptide and CDR loops". Two caveats belong on the figure, not in a footnote:
#
#   1. The `plddt` we hold is the GLOBAL pLDDT of the retained model, not the winner's
#      peptide-and-CDR-loop average. This plots a related quantity, not their statistic.
#   2. The between-set offsets are NOT a confound to be apologised for -- they are a property of
#      the scorer, and they are the point. A confidence metric offered as a binder/non-binder
#      classifier has to carry a TRANSFERABLE scale: if its distribution moves between datasets
#      the same tool generated, then no operating point transfers, and the thing only works
#      within a batch. This figure is a transferability test, not a confounded comparison.
#
#      The benchmark's own metric cannot see this. Per-peptide AUC is rank-based within a peptide,
#      so it is invariant to any per-peptide or per-batch shift -- a score can post macro-AUC_0.1
#      0.60 while having no threshold that survives a change of dataset. The quantity that exposes
#      it is the gap between the WITHIN-epitope AUC and the POOLED AUC under one global threshold,
#      computed below.
#
# The classes are NOT interchangeable and the figure keeps them apart:
#   TCRvdb              assay-validated (DESeq2 p_adj < 1e-5) binders and non-binders
#   vdjdb_binder_bench  real VDJdb binders vs MOCK SHUFFLED negatives
#   vdjdb_free_pool     deposited pairings vs MISPAIRINGS
#   immrep23_negatives  negatives ONLY (no contrast available)
#   IMMREP25            positives ONLY -- no negatives exist for its 20 peptides in any
#                       structure set we hold, which is itself the asymmetry worth seeing
#
# Run: python src/plddt_panels.py
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.expanduser("~/hf/tcren_structures")
RESULTS = os.path.join(REPO, "results")
FIGDIR = os.path.join(RESULTS, "figures")
ADAT = os.path.join(REPO, "appendix", "analysis")

BLUE, RED, GREY = "#2c7fb8", "#d7301f", "#8a8a8a"      # as iptm_compare.py
TEAL = "#41b6c4"
COLS = ["source", "epitope", "hla", "y", "plddt"]   # the one true column order (see load_immrep25)
MIN_CLASS = 15        # a box needs this many structures per class to be drawn
MAX_EPITOPES = 12     # panel B cap, for legibility; the drop is logged, never silent

SETS = [
    ("TCRvdb (assay-labelled)", "tcrvdb/metadata.tsv", "y", "epitope"),
    ("VDJdb benchmark (mock negatives)", "vdjdb_binder_benchmark/metadata.tsv", "y", "epitope"),
    ("VDJdb free pool (mispairings)", "vdjdb_free_pool/metadata.tsv", "y", "epitope"),
    ("IMMREP mismatched decoys", "immrep23_negatives/immrep2022_negatives.tsv", "label", "epitope"),
]


def load_immrep25() -> pl.DataFrame:
    t = pl.read_csv(os.path.join(RESULTS, "iptm_templates.csv"))
    assert t.height == 1000 and t["peptide"].n_unique() == 20, "unexpected IMMREP25 shape"
    # column ORDER is pinned here and in load_reference: pl.concat is positional by default, and
    # a mismatch would silently vstack non-binder values into a binder column
    return (t.select(epitope="peptide", plddt="plddt", hla="hla")
            .with_columns(source=pl.lit("IMMREP25"), y=pl.lit(1, pl.Int64))
            .select(COLS))


def load_reference() -> pl.DataFrame:
    frames = []
    for label, path, ycol, epcol in SETS:
        p = os.path.join(HF, path)
        if not os.path.exists(p):
            print("MISSING %s -- skipped" % p)
            continue
        t = pl.read_csv(p, separator="\t", infer_schema_length=10000)
        if "plddt" not in t.columns:
            print("no plddt in %s -- skipped" % label)
            continue
        t = (t.select(epitope=pl.col(epcol).cast(pl.Utf8), plddt=pl.col("plddt").cast(pl.Float64),
                      y=pl.col(ycol).cast(pl.Int64))
             .drop_nulls(["plddt", "epitope", "y"])
             .with_columns(source=pl.lit(label), hla=pl.lit(""))
             .select(COLS))
        print("%-34s %5d structures | %2d epitopes | classes %s"
              % (label, t.height, t["epitope"].n_unique(),
                 sorted(set(t["y"].to_list()))))
        frames.append(t)
    return pl.concat(frames)


def paired_epitopes(ref: pl.DataFrame) -> pl.DataFrame:
    """Epitopes with both classes at >= MIN_CLASS structures each, best-powered first."""
    g = (ref.group_by("source", "epitope", "y").agg(n=pl.len())
         .pivot(on="y", index=["source", "epitope"], values="n").fill_null(0))
    pos = "1" if "1" in g.columns else 1
    neg = "0" if "0" in g.columns else 0
    g = (g.rename({str(pos): "n_pos", str(neg): "n_neg"})
         .with_columns(n_min=pl.min_horizontal("n_pos", "n_neg"))
         .filter(pl.col("n_min") >= MIN_CLASS)
         .sort("n_min", descending=True))
    return g


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    imm = load_immrep25()
    ref = load_reference()

    # ---- per-peptide summary table, emitted so every box is checkable ----
    summ = (pl.concat([imm, ref])
            .group_by("source", "epitope", "y")
            .agg(n=pl.len(), median=pl.col("plddt").median(),
                 q1=pl.col("plddt").quantile(0.25), q3=pl.col("plddt").quantile(0.75),
                 lo=pl.col("plddt").min(), hi=pl.col("plddt").max())
            .sort("source", "epitope", "y"))
    summ.write_csv(os.path.join(RESULTS, "plddt_per_peptide.csv"))

    sel = paired_epitopes(ref)
    print("\nepitopes with both classes at >=%d each: %d" % (MIN_CLASS, sel.height))
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(sel)
    keep = sel.head(MAX_EPITOPES)
    if sel.height > MAX_EPITOPES:
        print("panel B caps at %d epitopes; DROPPED %d: %s"
              % (MAX_EPITOPES, sel.height - MAX_EPITOPES,
                 sel.tail(sel.height - MAX_EPITOPES)["epitope"].to_list()))

    # ---- transferability: within-epitope AUC vs one global threshold ----
    from sklearn.metrics import roc_auc_score
    print("\n=== is the pLDDT scale transferable? ===")
    within = []
    for r in sel.to_dicts():
        s = ref.filter((pl.col("source") == r["source"]) & (pl.col("epitope") == r["epitope"]))
        y, v = s["y"].to_numpy(), s["plddt"].to_numpy()
        within.append(roc_auc_score(y, v))
    print("within-epitope AUC, macro over %d epitopes: %.4f (sd %.4f, range %.4f-%.4f)"
          % (len(within), float(np.mean(within)), float(np.std(within, ddof=1)),
             min(within), max(within)))
    for src in sorted(set(ref["source"].to_list())):
        s = ref.filter(pl.col("source") == src)
        if s["y"].n_unique() < 2:
            print("  %-34s single class -- no AUC" % src)
            continue
        print("  %-34s pooled within source AUC %.4f (n=%d)"
              % (src, roc_auc_score(s["y"].to_numpy(), s["plddt"].to_numpy()), s.height))
    allc = ref.filter(pl.col("source") != "IMMREP mismatched decoys")
    pooled = roc_auc_score(allc["y"].to_numpy(), allc["plddt"].to_numpy())
    print("ONE GLOBAL THRESHOLD across all labelled sets: AUC %.4f (n=%d, %d binders)"
          % (pooled, allc.height, int(allc["y"].sum())))
    print("  -> the within-minus-pooled gap is %.4f: that much of the apparent performance is "
          "per-epitope recentring, not a usable operating point"
          % (float(np.mean(within)) - pooled))
    # persist the per-cohort values: the range across cohorts is the direct evidence that a
    # per-epitope score cannot rank methods, and it was previously stdout-only
    (sel.select("source", "epitope", "n_pos", "n_neg")
        .with_columns(within_auc=pl.Series(within))
        .sort("within_auc")
        .write_csv(os.path.join(RESULTS, "plddt_transfer.csv")))
    # where do IMMREP25's binders sit on the same scale?
    print("\nbinder median pLDDT by set (a transferable score would put these together):")
    for src in ["IMMREP25"] + sorted(set(ref["source"].to_list())):
        s = (imm if src == "IMMREP25" else ref.filter(pl.col("source") == src)).filter(
            pl.col("y") == 1)
        if s.height:
            print("  %-34s %.2f  (n=%d)" % (src, float(s["plddt"].median()), s.height))
    nb = ref.filter(pl.col("y") == 0)["plddt"].to_numpy()
    print("non-binder median across all sets: %.2f" % float(np.median(nb)))
    print("IMMREP25 true binders scoring BELOW that non-binder median: %.1f%%"
          % (100 * float((imm["plddt"].to_numpy() < np.median(nb)).mean())))

    # ---- macros. These are FULL AUCs, unlike every other AUC in this repo, because a
    # transferable operating point is a question about the whole ROC and not about the
    # low-false-positive region the benchmark integrates over. Labelled as such in the text.
    macros = {
        "pldNcohort": "%d" % len(within),
        "pldMinClass": "%d" % MIN_CLASS,
        "pldWithin": "%.4f" % float(np.mean(within)),
        "pldWithinSd": "%.4f" % float(np.std(within, ddof=1)),
        "pldWithinLo": "%.4f" % float(min(within)),
        "pldWithinHi": "%.4f" % float(max(within)),
        "pldPooled": "%.4f" % pooled,
        "pldPooledN": "%d" % allc.height,
        "pldGap": "%.4f" % (float(np.mean(within)) - pooled),
        "pldImmMedian": "%.2f" % float(imm["plddt"].median()),
        "pldNegMedian": "%.2f" % float(np.median(nb)),
        "pldImmBelowNeg": "%.1f" % (100 * float((imm["plddt"].to_numpy() < np.median(nb)).mean())),
        "pldNimm": "%d" % imm.height,
        "pldNpep": "%d" % imm["epitope"].n_unique(),
    }
    with open(os.path.join(ADAT, "plddt_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("wrote results/plddt_transfer.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "plddt_macros.tex"), len(macros)))

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.16)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    imm_med = float(imm["plddt"].median())

    # ---- Panel A: IMMREP25, one box per peptide, positives only ----
    order = (imm.group_by("epitope", "hla").agg(m=pl.col("plddt").median())
             .sort("m", descending=True))
    data, cols, labs = [], [], []
    for r in order.to_dicts():
        v = imm.filter(pl.col("epitope") == r["epitope"])["plddt"].to_numpy()
        data.append(v)
        cols.append(BLUE if r["hla"] == "A*02:01" else TEAL)
        labs.append("%s (%d)" % (r["epitope"], len(v)))
    bp = axA.boxplot(data, patch_artist=True, widths=0.62, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    axA.axhline(imm_med, ls="--", lw=0.9, color=GREY, zorder=0)
    axA.set_xticks(range(1, len(labs) + 1))
    axA.set_xticklabels(labs, rotation=90, fontsize=7)
    axA.set_ylabel("global pLDDT of the retained model")
    axA.set_title("IMMREP25 cognate complexes (no non-binders exist)", fontsize=9)
    axA.plot([], [], "s", color=BLUE, label="HLA-A*02:01")
    axA.plot([], [], "s", color=TEAL, label="HLA-B*40:01")
    axA.legend(fontsize=7, frameon=False, loc="lower left")

    # ---- Panel B: reference epitopes, binder vs non-binder ----
    pos_h, neg_h, ticks, tlabs, seps = [], [], [], [], []
    x = 1
    for r in keep.to_dicts():
        s = ref.filter((pl.col("source") == r["source"]) & (pl.col("epitope") == r["epitope"]))
        p = s.filter(pl.col("y") == 1)["plddt"].to_numpy()
        n = s.filter(pl.col("y") == 0)["plddt"].to_numpy()
        pos_h.append((x - 0.18, p)); neg_h.append((x + 0.18, n))
        ticks.append(x)
        assay = "*" if r["source"].startswith("TCRvdb") else ""
        tlabs.append("%s%s (%d/%d)" % (r["epitope"], assay, len(p), len(n)))
        x += 1
    for xs, arr, c in [(pos_h, None, BLUE), (neg_h, None, RED)]:
        for xi, v in xs:
            b = axB.boxplot([v], positions=[xi], widths=0.3, patch_artist=True, showfliers=False)
            b["boxes"][0].set_facecolor(c); b["boxes"][0].set_alpha(0.75)
            b["boxes"][0].set_edgecolor("black"); b["medians"][0].set_color("black")
    axB.axhline(imm_med, ls="--", lw=0.9, color=GREY, zorder=0)
    axB.set_xticks(ticks); axB.set_xticklabels(tlabs, rotation=90, fontsize=7)
    axB.set_title("Reference sets with both classes (binder / non-binder)", fontsize=9)
    axB.plot([], [], "s", color=BLUE, label="binder")
    axB.plot([], [], "s", color=RED, label="non-binder")
    axB.plot([], [], ls="--", color=GREY, label="IMMREP25 median (%.1f)" % imm_med)
    axB.legend(fontsize=7, frameon=False, loc="lower left")

    for ax, tag in ((axA, "a"), (axB, "b")):
        ax.text(-0.06, 1.06, tag, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="top", ha="left")
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
    fig.text(0.005, 0.005, "* assay-labelled (DESeq2 p_adj < 1e-5); all other non-binders are "
             "mock shuffles or mispairings. A confidence score used as a classifier needs one "
             "transferable operating point: if binder boxes do not align across sets, no single "
             "threshold works, and the per-peptide AUC the benchmark reports cannot detect it.",
             fontsize=6.5, color="#444444")

    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, "fig_plddt_panels." + ext)
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote %s" % out)
    print("wrote %s" % os.path.join(RESULTS, "plddt_per_peptide.csv"))
    print("\nIMMREP25 pooled median pLDDT %.2f (n=%d over %d peptides)"
          % (imm_med, imm.height, imm["epitope"].n_unique()))


if __name__ == "__main__":
    main()
