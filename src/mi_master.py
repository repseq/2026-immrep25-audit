"""Master mutual-information table: I(epitope; representation) in bits and normalized U=I/H(A),
per cohort, on ONE comparable scale across representations of the TCR.
# 2026-08-06

For every sequence cohort, on each chain separately, we put four representations of a receptor on the
same bits/U scale. These are four DIFFERENT lower-bound estimators of I(A; receptor) -- not a strict
nested chain -- and the point is that they AGREE on the cohort ordering:

  gene-usage (V,J)          discrete plug-in + permutation-excess (src/mi.mi_report)
  ESM Leiden cluster        I(A; cluster)         -- clustering-based (coarse-grains the embedding)
  ESM embedding (KSG)       I(A; embedding)       -- Ross 2014 kNN on PCA-reduced vectors; geometry-based
  held-out learning         I_heldout(A; pred)    -- the GENERALIZABLE bits (from degenerate.py)

U = excess bits / H(A): the fraction of epitope entropy the representation pins down, in excess of a
label-permutation null, so U=0 is chance for every representation and the four columns are one
quantity. IMMREP25 sits near zero on the three embedding/learning estimators (cluster, KSG,
held-out), with the verified cohorts well above. On gene usage it does NOT sit at the floor: it
carries about as much germline V/J information as the verified cohorts, which is the same fact the
germline-only benchmark baseline reports as a competitive AUC. (Normalising Miller-Madow rather than
excess bits instead reads U~0.5 for the antigen-blind controls -- that is finite-N bias on a sparse
V/J table, not donor structure.) The structural probe lives on a different, predicted-structure
cohort set and is reported separately.

Usage: uv run python src/mi_master.py   # writes results/mi_master.csv + .dat + macros
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META      # noqa: E402
from esm_signal import _subsample, CACHE, PCA_DIM              # noqa: E402
from esm_pca import project                                     # noqa: E402
from mi import mi_report, mi_ross_report, entropy_bits          # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
TABLES = os.path.join(os.path.dirname(REPO), "2026-immrep25-audit-ms", "tables")
CHAINS = ("A", "B")
MACRO_CHAIN = "B"     # the chain the in-text macros refer to
KSG_K = 3             # Ross's own recommendation and scikit-learn's default (see mi.mi_ross)
N_PERM = 200


def _from_csv(fname, cohort, chain, col):
    df = pd.read_csv(os.path.join(RESULTS, fname))
    m = (df.cohort == cohort) & (df.chain == chain) if "chain" in df else (df.cohort == cohort)
    r = df[m]
    return float(r[col].iloc[0]) if len(r) else np.nan


def run():
    rng = np.random.default_rng(0)
    coh = build_cohorts(include_olga=True)
    order = [c for c in HIERARCHY if c in coh]
    rows = []
    for name in order:
        long = coh[name]["long"]
        for chain in CHAINS:
            lc = long[long.chain == chain]
            keepcols = [c for c in ["epitope", "cdr3", "v", "j"] if c in lc.columns]
            sub = _subsample(lc[keepcols].dropna(subset=["epitope", "cdr3"]), rng)
            if sub.epitope.nunique() < 2:
                continue
            yt = pd.Categorical(sub.epitope).codes.astype(np.int64)
            # Every U below is permutation-EXCESS bits / H(A) -- see mi.mi_report -- so the column
            # is one quantity with one chance level (U=0) across all four representations.
            Ha = entropy_bits(yt)          # > 0: at least two epitopes qualify

            # gene-usage (V,J) -- discrete
            rep = {}
            if {"v", "j"}.issubset(sub.columns):
                gene = (sub.v.astype(str) + "_" + sub.j.astype(str)).to_numpy()
                g = mi_report(yt, pd.Categorical(gene).codes, n_perm=500)
                rep["gene"] = (g["excess"], g["U"], g["p"])

            # ESM embedding -- KSG/Ross on the GLOBAL PCA projection (src/esm_pca.py), so every
            # cohort is scored on the same axes rather than on axes fit to itself
            cache = os.path.join(CACHE, f"{name}_{chain}.npy")
            if os.path.exists(cache):
                X = np.load(cache)
                if X.shape[0] == len(sub):
                    r = mi_ross_report(project(X, chain), yt, k=KSG_K, n_perm=N_PERM)
                    rep["ksg"] = (r["excess"], r["U"], r["p"])

            # ESM Leiden cluster + held-out learning -- excess bits from the committed CSVs,
            # normalised here rather than read back, so this column cannot inherit a different
            # convention from whichever script last wrote them
            cl = _from_csv("esm_cluster.csv", name, chain, "mi_bits")
            rep["cluster"] = (cl, cl / Ha, _from_csv("esm_cluster.csv", name, chain, "mi_p"))
            hd = _from_csv("degenerate.csv", name, chain, "I_heldout")
            rep["held"] = (hd, hd / Ha, np.nan)

            for r_name, (I, U, p) in rep.items():
                rows.append({"cohort": name, "label": COHORT_META[name]["label"], "chain": chain,
                             "representation": r_name, "I_bits": I, "U": U, "p": p})
            print("%-16s TR%s  gene U=%.3f  cluster U=%.3f  ESM-KSG U=%.3f  held-out U=%.3f"
                  % (name, chain, rep.get("gene", (0, np.nan))[1], rep["cluster"][1],
                     rep.get("ksg", (0, np.nan))[1], rep["held"][1]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "mi_master.csv"), index=False)

    # wide .dat per chain (one row per cohort, one column per representation)
    idx = {n: i for i, n in enumerate(HIERARCHY)}
    reps = ["gene", "cluster", "ksg", "held"]
    for chain in CHAINS:
        wide = out[out.chain == chain].pivot_table(index="cohort", columns="representation",
                                                   values="U")
        with open(os.path.join(ADAT, "mi_master_%s.dat" % chain), "w") as fh:
            fh.write("# x cohort " + " ".join(reps) + "\n")
            for name in order:
                if name not in wide.index:
                    continue
                fh.write("%d %s %s\n" % (idx[name], name,
                         " ".join("%.4f" % wide.loc[name].get(r, np.nan) for r in reps)))

    # macros: the in-text values, TCRbeta (MACRO_CHAIN)
    macros = {}
    keymap = {"immrep25_pos": "Imm", "vdjdb_hq": "Hq", "immrep22_true": "Ii",
              "tcrvdb_true": "Tt", "vdjdb_lq": "Lq", "olga_random": "Olga"}
    mb = out[out.chain == MACRO_CHAIN]
    for cohort, tag in keymap.items():
        for rep_name, mtag in (("ksg", "Ksg"), ("gene", "Gene"), ("held", "Held")):
            r = mb[(mb.cohort == cohort) & (mb.representation == rep_name)]
            if len(r):
                macros["miU%s%s" % (mtag, tag)] = "%.3f" % r.U.iloc[0]
                macros["miBits%s%s" % (mtag, tag)] = "%.2f" % r.I_bits.iloc[0]
    macros["miKsgK"] = "%d" % KSG_K
    macros["miPcaDim"] = "%d" % PCA_DIM
    with open(os.path.join(ADAT, "mi_master_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    # LaTeX table (T4): the two label-predicting estimators, one column group per chain. The two
    # representation-GEOMETRY estimators (ESM cluster, ESM k-NN) are the figure instead, so no
    # value appears in both places.
    cols = [("gene", "gene usage"), ("held", "held-out")]
    piv = {ch: out[out.chain == ch].pivot_table(index="cohort", columns="representation",
                                                values="U") for ch in CHAINS}
    best = {(ch, c): piv[ch][c].max() for ch in CHAINS for c, _ in cols if c in piv[ch]}
    with open(os.path.join(TABLES, "mi_master_table.tex"), "w") as fh:
        fh.write("%% GENERATED by src/mi_master.py -- do not edit.\n"
                 "%% Regenerate: python src/mi_master.py (from the 2026-immrep25-audit repo).\n")
        fh.write("\\begin{tabular}{l" +
                 "@{\\hskip 1.5em}".join(["r" * len(cols)] * len(CHAINS)) + "}\n\\toprule\n")
        fh.write(" & " + " & ".join("\\multicolumn{%d}{c}{TCR$\\%s$}"
                                    % (len(cols), "alpha" if ch == "A" else "beta")
                                    for ch in CHAINS) + " \\\\\n")
        fh.write("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n"
                 % (1 + len(cols), 2 + len(cols), 1 + 2 * len(cols)))
        fh.write("cohort & " + " & ".join(h for _ in CHAINS for _, h in cols) + " \\\\\n\\midrule\n")
        for name in order:
            if not any(name in piv[ch].index for ch in CHAINS):
                continue
            cells = []
            for ch in CHAINS:
                for c, _ in cols:
                    v = piv[ch].loc[name].get(c, np.nan) if name in piv[ch].index else np.nan
                    if not np.isfinite(v):
                        cells.append("--"); continue
                    v += 0.0                                  # kill "-0.000"
                    s = ("$-$%.3f" % abs(v)) if round(v, 3) < 0 else "%.3f" % v
                    cells.append("\\textbf{%s}" % s
                                 if abs(v - best.get((ch, c), np.nan)) < 1e-9 else s)
            fh.write("%s & %s \\\\\n" % (COHORT_META[name]["tex"], " & ".join(cells)))
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nwrote results/mi_master.csv + mi_master_{A,B}.dat + macros + mi_master_table.tex")


if __name__ == "__main__":
    run()
