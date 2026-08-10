import marimo

__generated_with = "0.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    return mo, np, os, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Validation efficiency → benchmark usability

        A label-level argument, independent of any convergence probe. If a fraction *f* of the positive
        assignments are assay false positives, epitope signal is diluted by (1−*f*), and the
        between-method AUC spread contracts by (1−*f*). A benchmark can only **rank** de-novo predictors
        while that contracted spread exceeds the per-epitope AUC sampling noise — giving a **minimum
        validation efficiency** (1−*f*). With *f* ≈ 0.5 (Messemaker et al. 2025) IMMREP25 falls below it.

        Regenerate: `uv run python src/validation_efficiency.py`.
        """
    )
    return


@app.cell
def _(np, os):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dat = np.loadtxt(os.path.join(REPO, "appendix", "analysis", "valideff.dat"))
    macros = {}
    for line in open(os.path.join(REPO, "appendix", "analysis", "valideff_macros.tex")):
        line = line.strip()
        if line.startswith("\\newcommand"):
            k = line.split("{\\")[1].split("}")[0]
            v = line.rsplit("{", 1)[1].rstrip("}")
            macros[k] = v
    return dat, macros


@app.cell
def _(dat, macros, np, plt):
    f, retained, spread, thr = dat.T
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(f, spread, color="#2166ac", label="between-method AUC spread  (1−f)·Δ")
    ax.axhline(thr[0], color="#b2182b", ls="--", label="ranking noise  z·σ")
    fstar = float(macros["veFstar"]); fref = float(macros["veF"])
    ax.axvline(fstar, color="#b2182b", ls=":", label=f"f* = {fstar:.2f} (min. eff. {macros['veMinEff']}%)")
    ax.axvline(fref, color="k", ls=":", label=f"IMMREP25 f ≈ {fref:.2f}")
    ax.set_xlabel("false-positive fraction f"); ax.set_ylabel("AUC units")
    ax.legend(fontsize=8); fig.tight_layout()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(macros, mo):
    mo.md(
        f"**f\\* = {macros['veFstar']}** (minimum validation efficiency **{macros['veMinEff']}%**); "
        f"at f ≈ {macros['veF']} the benchmark is **{'unresolvable' if macros['veResolvable']=='no' else 'resolvable'}** "
        f"(σ(AUC, n={macros['veNpos']}) = {macros['veSigma']})."
    )
    return


if __name__ == "__main__":
    app.run()
