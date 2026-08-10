#!/usr/bin/env python
# 2026-08-06  What does the IMMREP25 benchmark actually reward?
#
# IMMREP25 forms negatives by re-using each TCR against the other 9 peptides OF ITS OWN MHC
# (Noakes et al.; within-MHC swaps). So the per-peptide task is really "which same-MHC MIRA pool did
# this receptor come from" -- and MIRA pools are clonal expansions that share V/J germline. We test how
# far that gets WITHOUT any epitope-specific CDR3 motif: a logistic regression on V/J gene usage +
# CDR3 length only (no CDR3 sequence), on the exact per-peptide one-vs-(same-MHC-rest) task, 5-fold CV.
#
# This is an IN-SAMPLE upper bound on the exploitable non-CDR3 signal; if even here the CDR3 sequence
# adds nothing over germline, no method -- structural or otherwise -- can be extracting CDR3-level
# recognition. Metric: macro AUC and macro AUC_0.1 (McClish partial AUC at FPR<=0.1), the benchmark's
# read-out.  Run: python src/germline_baseline.py
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dump" / "immrep25" / "immrep2025_for_release.tsv"
OUT_CSV = REPO / "results" / "germline_baseline.csv"
OUT_ROC = REPO / "results" / "germline_roc.csv"
MACROS = REPO / "appendix" / "analysis" / "germline_macros.tex"
FIG = REPO.parent / "2026-immrep25-audit-ms" / "figures" / "fig_germline.pdf"
AA = list("ACDEFGHIKLMNPQRSTVWY")
COMPETITOR = 0.60  # best of 126 IMMREP25 submissions, macro AUC_0.1 (Richardson 2026)


def aacomp(s):
    s = str(s); n = max(len(s), 1)
    return [s.count(a) / n for a in AA]


def germline(df):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    genes = oh.fit_transform(df[["tcra_v", "tcra_j", "tcrb_v", "tcrb_j"]].astype(str))
    lens = df[["tcra_cdr3", "tcrb_cdr3"]].apply(lambda c: c.str.len()).values
    return np.hstack([genes, lens])


def germline_cdr3(df):
    comp = np.array([aacomp(a) + aacomp(b) for a, b in zip(df.tcra_cdr3, df.tcrb_cdr3)])
    return np.hstack([germline(df), comp])


def evaluate(d, feat):
    full, p01 = [], []
    for E in sorted(d.peptide.unique()):
        sub = d[d.peptide == E].reset_index(drop=True)
        y = sub.label.values
        proba = cross_val_predict(LogisticRegression(max_iter=2000), feat(sub), y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  method="predict_proba")[:, 1]
        full.append(roc_auc_score(y, proba)); p01.append(roc_auc_score(y, proba, max_fpr=0.1))
    return np.array(full), np.array(p01)


def lopo_unseen(d):
    """Rigorous UNSEEN baseline: leave-one-peptide-out, germline + peptide features, no CDR3 motif.
    Trains on 19 peptides (learning peptide->germline preference), predicts the held-out peptide."""
    dd = d.copy()
    for a in AA:
        dd[f"p_{a}"] = dd.peptide.apply(lambda s: s.count(a) / len(s))
    dd["p_len"] = dd.peptide.str.len(); dd["la"] = dd.tcra_cdr3.str.len(); dd["lb"] = dd.tcrb_cdr3.str.len()
    cat = ["tcra_v", "tcra_j", "tcrb_v", "tcrb_j", "hla"]
    for c in cat:
        dd[c] = dd[c].astype("category")
    feats = cat + [f"p_{a}" for a in AA] + ["p_len", "la", "lb"]
    p01, full = [], []
    for E in sorted(dd.peptide.unique()):
        tr, te = dd[dd.peptide != E], dd[dd.peptide == E]
        m = HistGradientBoostingClassifier(categorical_features=cat, max_iter=300,
                                           learning_rate=0.05, random_state=0)
        m.fit(tr[feats], tr.label)
        s = m.predict_proba(te[feats])[:, 1]
        p01.append(roc_auc_score(te.label, s, max_fpr=0.1))
        # Full AUC too: the negatives of peptide E are the POSITIVES of the other nine same-MHC
        # peptides, so a model that learned their germline signatures ranks them above E's own
        # positives. AUC_0.1 saturates at its floor here and hides that; the full AUC does not.
        full.append(roc_auc_score(te.label, s))
    return np.array(p01), np.array(full)


def _std_pauc01_binormal(dprime, p=0.1):
    """McClish (1989)-standardised partial AUC over FPR<=p of the equal-variance binormal ROC
    TPR=Phi(Phi^-1(fpr)+dprime); matches sklearn roc_auc_score(max_fpr=p). chance->0.5, perfect->1."""
    area, _ = quad(lambda x: norm.cdf(norm.ppf(x) + dprime), 1e-9, p, limit=200)
    lo, hi = 0.5 * p * p, p            # area under chance / perfect ROC over [0,p]
    return 0.5 * (1 + (area - lo) / (hi - lo))


def roc_curves(d):
    """Per-epitope germline-only ROC curves on a common FPR grid -> OUT_ROC (long format).

    The curves are composed into Fig. 4b by the manuscript repo's make_figs.py, so this only
    writes the data. Recomputes the germline CV probabilities (cheap) so evaluate()'s signature
    stays unchanged. Rows: series = one peptide, 'macro' (the epitope-averaged ROC) or 'binormal'
    (the reference below). hla is carried through so the panel can colour curves by restricting
    allele -- the negatives are same-MHC, so the two alleles are the two competing pools."""
    peps = sorted(d.peptide.unique())
    pep2hla = d.groupby("peptide").hla.first().to_dict()
    grid = np.linspace(0, 1, 201)
    rows, tprs, aucs01 = [], [], []
    for E in peps:
        sub = d[d.peptide == E].reset_index(drop=True)
        y = sub.label.values
        proba = cross_val_predict(LogisticRegression(max_iter=2000), germline(sub), y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  method="predict_proba")[:, 1]
        tpr_i = np.interp(grid, *roc_curve(y, proba)[:2])
        tprs.append(tpr_i); aucs01.append(roc_auc_score(y, proba, max_fpr=0.1))
        rows.append(pd.DataFrame({"series": E, "hla": pep2hla[E], "fpr": grid, "tpr": tpr_i,
                                  "auc01": roc_auc_score(y, proba, max_fpr=0.1)}))
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[0] = 0.0
    rows.append(pd.DataFrame({"series": "macro", "hla": "", "fpr": grid, "tpr": mean_tpr,
                              "auc01": float(np.mean(aucs01))}))
    # Theoretical reference: the equal-variance binormal ROC (the maximum-likelihood smooth-ROC
    # model, Dorfman & Alf 1969), its single parameter d' set so its McClish-standardised partial
    # AUC over FPR<=0.1 equals the reported best competitor score (0.60). TPR = Phi(Phi^-1(fpr)+d').
    dprime = brentq(lambda dd: _std_pauc01_binormal(dd) - COMPETITOR, 0.0, 5.0)
    ref = norm.cdf(norm.ppf(np.clip(grid, 1e-9, 1 - 1e-9)) + dprime); ref[0] = 0.0
    rows.append(pd.DataFrame({"series": "binormal", "hla": "", "fpr": grid, "tpr": ref,
                              "auc01": COMPETITOR}))
    pd.concat(rows, ignore_index=True).to_csv(OUT_ROC, index=False)


def main():
    d = pd.read_csv(DATA, sep="\t")
    assert d.label.sum() == 1000 and len(d) == 10000, "unexpected benchmark shape"
    g_full, g_p01 = evaluate(d, germline)
    c_full, c_p01 = evaluate(d, germline_cdr3)
    lopo_p01, lopo_full = lopo_unseen(d)

    peps = sorted(d.peptide.unique())
    pd.DataFrame({"peptide": peps, "germline_auc": g_full, "germline_auc01": g_p01,
                  "germline_cdr3_auc": c_full, "germline_cdr3_auc01": c_p01,
                  "lopo_unseen_auc01": lopo_p01, "lopo_unseen_auc": lopo_full}).to_csv(OUT_CSV, index=False)

    def ci95(v):
        se = v.std(ddof=1) / np.sqrt(len(v))
        return v.mean() - 1.96 * se, v.mean() + 1.96 * se
    g_lo, g_hi = ci95(g_p01)
    l_lo, l_hi = ci95(lopo_full)
    # macro-AUC_0.1 saturates at its lower bound when no positive ranks inside FPR<=p: the partial
    # area is 0, and McClish standardisation 0.5*(1 + (0 - p^2/2)/(p - p^2/2)) collapses to
    # 0.5*(1 - (p/2)/(1 - p/2)) = 9/19 = 0.474 at p=0.1. That is the metric's floor, not chance (0.5).
    _p = 0.1
    p01_floor = 0.5 * (1 - (_p / 2) / (1 - _p / 2))
    n_at_floor = int(np.sum(np.isclose(lopo_p01, p01_floor, atol=1e-6)))
    macros = [
        (r"\germAuc", f"{g_full.mean():.2f}"), (r"\germAucP", f"{g_p01.mean():.2f}"),
        (r"\germCdrAucP", f"{c_p01.mean():.2f}"),
        (r"\germAucPlo", f"{g_p01.min():.2f}"), (r"\germAucPhi", f"{g_p01.max():.2f}"),
        (r"\germAucPciLo", f"{g_lo:.2f}"), (r"\germAucPciHi", f"{g_hi:.2f}"),
        (r"\germNabove", f"{int((g_p01 >= COMPETITOR).sum())}"),
        (r"\germNtotal", f"{len(peps)}"),
        (r"\germCdrGain", f"{c_p01.mean() - g_p01.mean():.3f}"),
        (r"\germLopoP", f"{lopo_p01.mean():.2f}"),
        (r"\germLopoPfloor", f"{p01_floor:.2f}"), (r"\germLopoNfloor", f"{n_at_floor}"),
        (r"\germLopoAuc", f"{lopo_full.mean():.2f}"),
        (r"\germLopoAucLo", f"{l_lo:.2f}"), (r"\germLopoAucHi", f"{l_hi:.2f}"),
    ]
    MACROS.write_text("".join(f"\\newcommand{{{k}}}{{{v}}}\n" for k, v in macros))

    # supplement figure: per-peptide germline AUC_0.1 vs the competitor ceiling
    order = np.argsort(g_p01)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(peps))
    ax.bar(x - 0.2, g_p01[order], 0.4, color="#2c7fb8", label="germline (V/J + CDR3 length)")
    ax.bar(x + 0.2, c_p01[order], 0.4, color="#d7301f", alpha=0.7, label="+ full CDR3 sequence")
    ax.axhline(COMPETITOR, ls="--", c="black", lw=1, label=f"best of 126 submissions ({COMPETITOR})")
    ax.axhline(g_p01.mean(), ls=":", c="#2c7fb8", lw=1)
    ax.axhline(0.5, ls="-", c="grey", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([peps[i] for i in order], rotation=90, fontsize=6)
    # colour each peptide label by its restricting HLA (IMMREP25 uses two alleles)
    pep2hla = d.groupby("peptide").hla.first().to_dict()
    hla_col = {"A*02:01": "#1b7837", "B*40:01": "#762a83"}
    for tick, i in zip(ax.get_xticklabels(), order):
        tick.set_color(hla_col.get(pep2hla.get(peps[i]), "black"))
    ax.set_ylabel("macro AUC$_{0.1}$"); ax.set_ylim(0.5, 0.85)   # start at chance
    ax.set_title("Benchmark reproduced by germline usage alone (no CDR3 motif, no structure)", fontsize=8)
    leg1 = ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    from matplotlib.patches import Patch
    hla_handles = [Patch(color=hla_col[h], label=f"HLA-{h}") for h in sorted(hla_col)]
    ax.add_artist(leg1)
    ax.legend(handles=hla_handles, title="peptide-label colour", fontsize=6, title_fontsize=6,
              loc="upper center", frameon=False)
    fig.tight_layout(); fig.savefig(FIG, bbox_inches="tight")
    roc_curves(d)

    print(f"germline (V/J + CDR3 length):   macro AUC={g_full.mean():.3f}  macro AUC_0.1={g_p01.mean():.3f}"
          f"  (range {g_p01.min():.2f}-{g_p01.max():.2f})")
    print(f"+ CDR3 aa-composition:          macro AUC={c_full.mean():.3f}  macro AUC_0.1={c_p01.mean():.3f}"
          f"  (CDR3 adds {c_p01.mean() - g_p01.mean():+.3f})")
    print(f"UNSEEN leave-one-peptide-out:   macro AUC_0.1={lopo_p01.mean():.3f} (saturated at floor {p01_floor:.3f} for {n_at_floor}/{len(peps)}); full AUC={lopo_full.mean():.3f} [{l_lo:.3f}, {l_hi:.3f}]")
    print(f"best of 126 submissions (Richardson 2026): macro AUC_0.1 = {COMPETITOR}")
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_ROC}\nwrote {MACROS}\nwrote {FIG}")


if __name__ == "__main__":
    main()
