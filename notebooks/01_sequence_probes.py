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
        # Model-free sequence probes

        Two biologically-grounded, training-free probes of whether a positive set is epitope-structured,
        scored per epitope (one-vs-many) and calibrated against the reference ladder + controls
        (`run_audit.py`):

        * **CDR3 homology S/N** — do same-epitope TCRs converge in CDR3? (background S/N = 1)
        * **Inter-chain pairing** — gene-usage bias and α/β mutual-information excess, in bits.

        Regenerate: `uv run --extra gen python run_audit.py` (needs the fetched cohorts).
        """
    )
    return


@app.cell
def _(load, mo, ordered):
    mo.md("### Homology S/N at d=1, TCRβ (background = 1)")
    hom = ordered(load("homology_sn.csv")[lambda d: (d.chain == "B") & (d.d == 1)])
    hom[["cohort", "sn", "lo", "hi", "n_ep"]]
    return


@app.cell
def _(load, mo, np, ordered, plt):
    mo.md("### Inter-chain pairing (bits): gene-usage bias vs α/β mutual information")
    pr = ordered(load("pairing.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(pr)); w = 0.38
    ax.bar(x - w / 2, pr.gene_mean, w, label="gene-usage bias", color="#762a83")
    ax.bar(x + w / 2, pr.mi_mean, w, label="inter-chain MI", color="#af8dc3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(pr.cohort, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("excess over null (bits)"); ax.legend(fontsize=8)
    fig.tight_layout()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
