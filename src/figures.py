"""Matplotlib figures: S/N hierarchy, homology graph layouts, pgen match, positive controls.

Reads results/*.csv (written by run_audit.py) and the live cohorts. Saves to
results/figures/. gnuplot/TikZ versions for the appendix live in appendix/analysis.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cohorts import build_cohorts, HIERARCHY, COHORT_META
from homology_graph import build_graph, sfdp_layout, graph_subsets

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(RESULTS, "figures"); os.makedirs(FIGS, exist_ok=True)

EXPECT_COLOR = {"signal": "#2166ac", "weak": "#67a9cf", "noise": "#b2182b",
                "test": "#ef8a00", "real": "#756bb1"}


def _load_cohorts():
    return build_cohorts(include_olga=True)


def fig_hierarchy():
    hom = pd.read_csv(os.path.join(RESULTS, "homology_sn.csv"))
    pair = pd.read_csv(os.path.join(RESULTS, "pairing.csv"))
    order = [c for c in HIERARCHY if c in set(hom.cohort)]
    labels = [COHORT_META[c]["label"] for c in order]
    colors = [EXPECT_COLOR[COHORT_META[c]["expect"]] for c in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, chain in zip(axes[:2], ("B", "A")):
        d1 = hom[(hom.chain == chain) & (hom.d == 1)].set_index("cohort").loc[order]
        x = np.arange(len(order))
        ax.bar(x, d1.sn, color=colors, alpha=0.5)
        ax.errorbar(x, d1.sn, yerr=[d1.sn - d1.lo, d1.hi - d1.sn],
                    fmt="o", color="k", capsize=3, zorder=3)
        ax.axhline(1, ls="--", c="grey", lw=1)
        ax.set_yscale("log"); ax.set_title(f"Homology S/N (TR{chain}, d=1)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    # pairing panel (excess bits)
    ax = axes[2]
    pr = pair.set_index("cohort").loc[[c for c in order if c in set(pair.cohort)]]
    x = np.arange(len(pr))
    ax.errorbar(x - 0.15, pr.gene_mean, yerr=[pr.gene_mean - pr.gene_lo, pr.gene_hi - pr.gene_mean],
                fmt="s", capsize=3, label="gene-usage bias", color="#1b7837")
    ax.errorbar(x + 0.15, pr.mi_mean, yerr=[pr.mi_mean - pr.mi_lo, pr.mi_hi - pr.mi_mean],
                fmt="^", capsize=3, label="inter-chain MI", color="#762a83")
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_title("Permutation-corrected pairing information (bits)")
    ax.set_xticks(x); ax.set_xticklabels([COHORT_META[c]["label"] for c in pr.index], rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_hierarchy.pdf")); plt.close(fig)
    print("wrote fig_hierarchy.pdf")


def fig_homology_graphs(coh, chain="B", min_n=30):
    # MLR is beta-only; AIRR random is dropped from the panels. OLGA random is kept.
    order = [c for c in HIERARCHY if c in coh and c not in ("mlr_prolif", "airr_control")]
    gsub = graph_subsets(coh, order, chain=chain, min_n=min_n)   # <= 50 receptors per epitope
    n = len(order); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, order):
        sub, full_n, n_eps = gsub[name]
        G = build_graph(sub)
        cats = {e: i for i, e in enumerate(sorted(set(sub.epitope)))}
        cmap = plt.cm.tab10
        deg = dict(G.degree())
        # Most of every cohort's receptors have no neighbour at all. Drawing them is worse than
        # useless: sfdp packs disconnected singletons into a regular lattice, so the panel fills
        # with a grid that is a layout artefact rather than data, and it hides the components the
        # panel exists to show. They are omitted here and counted in the title instead; the
        # layout runs on the connected subgraph alone, which is also far faster.
        con = [i for i in G.nodes if deg[i] > 0]
        H = G.subgraph(con)
        pos = sfdp_layout(H)
        if H.number_of_edges() and pos:                 # LineCollection: draws large graphs fast
            segs = [(pos[u], pos[v]) for u, v in H.edges if u in pos and v in pos]
            ax.add_collection(LineCollection(segs, colors="#999999", linewidths=0.3,
                                             alpha=0.7, zorder=2))
        drawn = [i for i in con if i in pos]
        if drawn:
            ax.scatter([pos[i][0] for i in drawn], [pos[i][1] for i in drawn],
                       s=[float(np.clip(5 + 3 * deg[i], 5, 90)) for i in drawn],
                       c=[cmap(cats[G.nodes[i]["epitope"]] % 10) for i in drawn],
                       alpha=0.9, linewidths=0, zorder=3)
        nn = G.number_of_nodes()
        ntxt = f"{nn} of {full_n}" if full_n > nn else f"{nn}"     # honest when subsampled
        ax.set_title(f"{COHORT_META[name]['tex']}\n$N$={ntxt} receptors, {len(cats)} epitopes, "
                     f"{G.number_of_edges()} edges,\n{nn - len(con)} with no neighbour (not drawn)",
                     fontsize=7.5, linespacing=1.35)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect(1)                              # square panels (aspect ratio 1)
    for ax in axes[n:]:
        ax.axis("off")
    # no suptitle: what the nodes, edges, colours and per-panel counts mean belongs in the legend
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"fig_homology_graph_TR{chain}.pdf")); plt.close(fig)
    print(f"wrote fig_homology_graph_TR{chain}.pdf")


def fig_pgen_match():
    cache = os.path.join(REPO, "cache", "immrep_pgen.npz")
    if not os.path.exists(cache):
        return
    d = np.load(cache); pa, pb = d["pa"], d["pb"]
    pool_b = pd.read_csv(os.path.join(REPO, "cache", "olga_pool_B.tsv"), sep="\t",
                         names=["cdr3", "v", "j", "pgen"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, arr, title in [(axes[0], pa, "TRA"), (axes[1], pb, "TRB")]:
        v = np.log10(arr[arr > 0])
        ax.hist(v, bins=40, density=True, alpha=0.6, label="IMMREP25 pos", color="#ef8a00")
        ax.set_title(f"log10 pgen ({title})"); ax.set_xlabel("log10 pgen")
    gb = pool_b[pool_b.pgen > 0]
    axes[1].hist(np.log10(gb.pgen), bins=40, density=True, alpha=0.4, label="OLGA raw gen", color="#4575b4")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_pgen.pdf")); plt.close(fig)
    print("wrote fig_pgen.pdf")


def fig_controls():
    """Independent positive controls: an orthogonal assay (10x dCODE dextramer) and a different
    MIRA dataset (ImmuneCODE MIRA-COVID), both scored with the audit's own homology probe. Shows
    that the probe detects convergence when the labels carry it -- so IMMREP25's position at
    background is a property of its labels, not of the probe or of the MIRA assay."""
    hom = pd.read_csv(os.path.join(RESULTS, "homology_sn.csv"))
    imm = float(hom[(hom.cohort == "immrep25_pos") & (hom.chain == "B") & (hom.d == 1)].sn.iloc[0])
    dc = pd.read_csv(os.path.join(RESULTS, "dcode.csv")).sort_values("sn1")
    mc = pd.read_csv(os.path.join(RESULTS, "mira_covid.csv")).sort_values("sn1", ascending=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # both chains: the probe is run per chain, and on this control TCRalpha is the stronger one.
    # Hollow bar = alpha, filled = beta; fill shade still marks the HLA-A*02:01 subset.
    a02 = dc.hla.astype(str).str.startswith("A*02:01")
    y = np.arange(len(dc))
    a1.barh(y + 0.20, dc.sn1, height=0.38, color=np.where(a02, "#2166ac", "#92c5de"),
            edgecolor="black", linewidth=0.4, label=r"TCR$\beta$")
    a1.barh(y - 0.20, dc.snA.fillna(0.0), height=0.38, facecolor="none",
            edgecolor=np.where(a02, "#2166ac", "#92c5de"), linewidth=1.0, label=r"TCR$\alpha$")
    a1.set_yticks(y); a1.set_yticklabels(dc.epitope, fontsize=7, family="monospace")
    a1.set_ylim(-0.7, len(dc) - 0.3)
    a1.set_xscale("log"); a1.set_xlabel("homology S/N ($d{=}1$)")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(fc="#2166ac", ec="black", lw=0.4, label=r"TCR$\beta$, HLA-A*02:01"),
                       Patch(fc="#92c5de", ec="black", lw=0.4, label=r"TCR$\beta$, other allele"),
                       Patch(fc="none", ec="#2166ac", lw=1.0, label=r"TCR$\alpha$ (same colouring)")],
              loc="lower right", fontsize=6.5, frameon=False)

    a2.plot(np.arange(1, len(mc) + 1), mc.sn1, "-", lw=1.6, color="#35978f")
    a2.set_yscale("log"); a2.set_xlabel("peptide pool, ranked by S/N")
    a2.set_ylabel("homology S/N (TCR$\\beta$, $d{=}1$)")

    for ax in (a1, a2):
        line = ax.axvline if ax is a1 else ax.axhline
        line(1.0, ls="--", c="#888888", lw=1, zorder=10)
        line(imm, ls="-.", c="#ef8a00", lw=1.4, zorder=10)
    # reference lines are labelled once, at the left bound of panel b; panel a carries none
    a2.set_xlim(left=0)
    a2.text(0.02, imm, "IMMREP25 ", color="#ef8a00", fontsize=7, ha="left", va="bottom",
            transform=a2.get_yaxis_transform())
    a2.text(0.02, 1.0, "background ", color="#888888", fontsize=7, ha="left", va="bottom",
            transform=a2.get_yaxis_transform())
    for ax, lab in ((a1, "a"), (a2, "b")):        # panel labels outside, top left
        ax.text(0.0, 1.02, "(%s)" % lab, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_controls.pdf")); plt.close(fig)
    print("wrote fig_controls.pdf")


if __name__ == "__main__":
    coh = _load_cohorts()
    fig_hierarchy()
    fig_homology_graphs(coh, "B")
    fig_homology_graphs(coh, "A")
    fig_pgen_match()
    fig_controls()
