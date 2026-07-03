"""Matplotlib figures: S/N hierarchy, homology graph layouts, UMAP density, pgen match.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cohorts import build_cohorts, HIERARCHY, COHORT_META
from homology_graph import build_graph, sfdp_layout, kmer_umap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(RESULTS, "figures"); os.makedirs(FIGS, exist_ok=True)

EXPECT_COLOR = {"signal": "#2166ac", "weak": "#67a9cf", "noise": "#b2182b", "test": "#ef8a00"}


def _load_cohorts():
    p = os.path.join(RESULTS, "olga_control.tsv")
    olga = pd.read_csv(p, sep="\t") if os.path.exists(p) else None
    return build_cohorts(include_olga=olga is not None, olga_paired=olga)


def fig_hierarchy():
    hom = pd.read_csv(os.path.join(RESULTS, "homology_sn.csv"))
    pair = pd.read_csv(os.path.join(RESULTS, "pairing.csv"))
    order = [c for c in HIERARCHY if c in set(hom.cohort)]
    labels = [COHORT_META[c]["label"] for c in order]
    colors = [EXPECT_COLOR[COHORT_META[c]["expect"]] for c in order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, chain in zip(axes[:2], ("B", "A")):
        d1 = hom[(hom.chain == chain) & (hom.d == 1)].set_index("cohort").loc[order]
        x = np.arange(len(order))
        ax.errorbar(x, d1.sn_median, yerr=[d1.sn_median - d1.sn_lo, d1.sn_hi - d1.sn_median],
                    fmt="o", color="k", capsize=3, zorder=3)
        ax.bar(x, d1.sn_median, color=colors, alpha=0.5)
        ax.axhline(1, ls="--", c="grey", lw=1)
        ax.set_yscale("log"); ax.set_title(f"Homology S/N (TR{chain}, d=1)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    # pairing panel
    ax = axes[2]
    pr = pair.set_index("cohort").loc[[c for c in order if c in set(pair.cohort)]]
    x = np.arange(len(pr))
    ax.errorbar(x - 0.15, pr.gene_bias_z, yerr=[pr.gene_bias_z - pr.gene_bias_z_lo, pr.gene_bias_z_hi - pr.gene_bias_z],
                fmt="s", capsize=3, label="gene-usage bias z", color="#1b7837")
    ax.errorbar(x + 0.15, pr.mi_z, yerr=[pr.mi_z - pr.mi_z_lo, pr.mi_z_hi - pr.mi_z],
                fmt="^", capsize=3, label="inter-chain MI z", color="#762a83")
    ax.axhline(0, ls="--", c="grey", lw=1); ax.axhline(2, ls=":", c="grey", lw=0.8)
    ax.set_title("Pairing non-randomness (z vs null)")
    ax.set_xticks(x); ax.set_xticklabels([COHORT_META[c]["label"] for c in pr.index], rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_hierarchy.pdf")); plt.close(fig)
    print("wrote fig_hierarchy.pdf")


def fig_homology_graphs(coh, chain="B", per_epi=60, n_epi=6):
    order = [c for c in HIERARCHY if c in coh]
    n = len(order); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, order):
        lc = coh[name]["long"]; lc = lc[lc.chain == chain]
        vc = lc.epitope.value_counts(); eps = vc[vc >= 15].index.to_numpy()
        if eps.size > n_epi:
            eps = rng.choice(eps, n_epi, replace=False)
        parts = [lc[lc.epitope == e].sample(min(per_epi, (lc.epitope == e).sum()), random_state=0) for e in eps]
        sub = pd.concat(parts, ignore_index=True) if len(parts) else lc.head(0)
        G = build_graph(sub); pos = sfdp_layout(G)
        cats = {e: i for i, e in enumerate(sorted(set(sub.epitope)))}
        cmap = plt.cm.tab10
        for u, v in G.edges:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], "-", color="#bbbbbb", lw=0.4, zorder=1)
        if pos:
            xs = [pos[i][0] for i in G.nodes]; ys = [pos[i][1] for i in G.nodes]
            cs = [cmap(cats[G.nodes[i]["epitope"]] % 10) for i in G.nodes]
            ax.scatter(xs, ys, s=9, c=cs, zorder=2, linewidths=0)
        ax.set_title(f"{COHORT_META[name]['label']}  (E={len(cats)}, edges={G.number_of_edges()})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Homology graph (TR{chain}): nodes=CDR3, edges=Hamming≤1, colour=epitope", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"fig_homology_graph_TR{chain}.pdf")); plt.close(fig)
    print(f"wrote fig_homology_graph_TR{chain}.pdf")


def fig_pgen_match():
    cache = os.path.join(REPO, "cache", "immrep_pgen.npz")
    if not os.path.exists(cache):
        return
    d = np.load(cache); pa, pb = d["pa"], d["pb"]
    pool_b = pd.read_csv(os.path.join(REPO, "cache", "olga_pool_b.tsv"), sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, arr, title in [(axes[0], pa, "TRA"), (axes[1], pb, "TRB")]:
        v = np.log10(arr[arr > 0])
        ax.hist(v, bins=40, density=True, alpha=0.6, label="immrep25 pos", color="#ef8a00")
        ax.set_title(f"log10 pgen ({title})"); ax.set_xlabel("log10 pgen")
    gb = pool_b[pool_b.pgen > 0]
    axes[1].hist(np.log10(gb.pgen), bins=40, density=True, alpha=0.4, label="OLGA raw gen", color="#4575b4")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_pgen.pdf")); plt.close(fig)
    print("wrote fig_pgen.pdf")


if __name__ == "__main__":
    coh = _load_cohorts()
    fig_hierarchy()
    fig_homology_graphs(coh, "B")
    fig_homology_graphs(coh, "A")
    fig_pgen_match()
