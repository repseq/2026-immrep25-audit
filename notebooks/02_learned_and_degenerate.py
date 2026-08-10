import marimo

__generated_with = "0.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _common import load, ordered
    return load, mo, np, ordered, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Learned representations & degenerate learning (headline)

        Two model-based probes on the cached ESM-2 embeddings (`cache/esm/*.npy`), no GPU needed:

        1. **ESM clustering** (`esm_cluster.py`) — Leiden partition vs epitope labels: chance-corrected
           ARI/AMI + MI in bits.
        2. **Degenerate-learning** (`degenerate.py`, the main result) — a linear probe is trained to
           predict the epitope from the embedding and evaluated on **held-out same-epitope TCRs**
           (grouped by CDR3, so held-out sequences are never training copies). A real recognition
           mechanism transfers (held-out AUC ≫ 0.5, normalized transfer *U* = I_heldout/H(A) > 0);
           degenerate memorization does not.

        Regenerate: `uv run python src/esm_cluster.py && uv run python src/degenerate.py`.
        """
    )
    return


@app.cell
def _(load, mo, ordered):
    deg = ordered(load("degenerate.csv")[lambda d: d.chain == "B"])
    mo.md("### Degenerate-learning (TCRβ): held-out transfer vs training memorization")
    return (deg,)


@app.cell
def _(deg):
    deg[["cohort", "n_ep", "train_auc", "heldout_auc", "gap",
         "I_train", "I_heldout", "U_heldout"]]
    return


@app.cell
def _(deg, plt):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = range(len(deg))
    ax.bar(x, deg.U_heldout, color=["#2166ac" if u > 0.1 else "#b2182b" for u in deg.U_heldout])
    ax.set_xticks(list(x)); ax.set_xticklabels(deg.cohort, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out transfer  U = I_heldout / H(A)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Normalized held-out transfer: gold sets learn a mechanism; IMMREP25 at background")
    fig.tight_layout()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(load, mo, ordered):
    mo.md("### ESM clustering (TCRβ): chance-corrected ARI/AMI + MI (bits)")
    esm = ordered(load("esm_cluster.csv")[lambda d: d.chain == "B"])
    esm[["cohort", "n", "n_clusters", "ari", "ami", "mi_bits", "mi_U"]]
    return


if __name__ == "__main__":
    app.run()
