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
    from _common import load
    return load, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Structural-convergence probe (caveated supplement)

        TCRen (Karnaukhov 2024) is used purely as an interface **feature extractor** over predicted
        complexes (`results/struct_desc_*.tsv`). Metrics are chance-corrected (ARI/AMI, MI-in-bits,
        within-cohort label-permutation background); the kNN S/N is a scaffold-inflated diagnostic only.

        **Instrument control (C1):** template-free *uncovered* VDJdb binders still separate strongly,
        so the probe has real power on the AlphaFold-Multimer pipeline; covered−uncovered is the
        template-leakage term. **IMMREP25 sits at its own scaffold-matched permutation background.**

        Regenerate: `uv run python src/structure_signal.py`.
        """
    )
    return


@app.cell
def _(load):
    sc = load("struct_convergence.csv")
    sc[["cohort", "n", "n_ep", "ari", "ami", "mi_bits", "mi_U", "perm_ari97", "sn", "sn_cdr3"]]
    return (sc,)


@app.cell
def _(np, plt, sc):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(sc)); w = 0.38
    ax.bar(x - w / 2, sc.ari, w, label="ARI", color="#2166ac")
    ax.bar(x + w / 2, sc.ami, w, label="AMI", color="#67a9cf")
    ax.plot(x, sc.perm_ari97, "k_", ms=14, label="perm. background (ARI 97.5%)")
    ax.set_xticks(x); ax.set_xticklabels(sc.cohort, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("chance-corrected index"); ax.legend(fontsize=8)
    ax.set_title("Structural epitope-convergence: real binders separate, IMMREP25 at background")
    fig.tight_layout()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
