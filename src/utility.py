#!/usr/bin/env python
# 2026-09-13  What does a macro-AUC_0.1 of 0.60 actually buy?
#
# The reviewers' substantive objection is about the SCOPE of a claim of absence, and the
# question it really turns on is what the benchmark's reported score would be worth if it
# were real. So instead of arguing about how little signal is there, we price it: two
# deployment tasks that the field actually cites as the motivation for TCR-pMHC prediction,
# evaluated at the benchmark's own score.
#
#   Task A  pull cognate TCRs out of a repertoire. Rank 10^6 receptors, take the top k, and
#           report expected true positives, precision, recall and enrichment at realistic
#           cognate precursor frequencies.
#   Task B  identify a TCR's epitope from a candidate panel of M. Report the probability the
#           cognate epitope ranks in the top 1/5/10 against the 1/M chance level.
#
# Both are computed twice: from a binormal ROC calibrated to a target macro-AUC_0.1, and
# from the germline classifier's EMPIRICAL per-peptide ROC curves (results/germline_roc.csv),
# so no conclusion rests on the binormal fit. Neither task needs any property of any method
# beyond its ROC, which is why this is a statement about the metric rather than about models.
#
# The load-bearing result is that the failure is not specific to 0.60: a hypothetical strong
# predictor at AUC_0.1 = 0.90 still returns ~1.5 true positives per 100 picks at 10^-5
# prevalence. macro-AUC_0.1 does not translate into deployment value anywhere in its usable
# range, which is a property of the metric under this class imbalance.
#
# Run: python src/utility.py
from __future__ import annotations
import os

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import binom, norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

AUC_BEST = 0.60          # best of 126 IMMREP25 submissions (Richardson 2026)
STRONG = 0.90            # a hypothetical strong predictor, for the range argument
MAX_FPR = 0.1            # the region the challenge metric integrates over
N_REP = 10 ** 6          # repertoire size for Task A
PREVALENCES = (1e-5, 1e-6)
TOPK = (100, 1000, 10000)
PANELS = (20, 100, 1000)


def std_pauc01(dprime: float, f: float = MAX_FPR) -> float:
    """McClish-standardised partial AUC over FPR<=f of the binormal ROC TPR=Phi(Phi^-1(x)+d').

    Matches sklearn's roc_auc_score(max_fpr=f): chance maps to 0.5, perfect to 1.
    """
    area, _ = quad(lambda x: norm.cdf(norm.ppf(x) + dprime), 1e-12, f, limit=300)
    lo, hi = 0.5 * f * f, f
    return 0.5 * (1 + (area - lo) / (hi - lo))


def calibrate(target: float) -> float:
    """The binormal separation d' whose standardised partial AUC equals `target`."""
    return float(brentq(lambda x: std_pauc01(x) - target, 0.0, 8.0))


def binormal_roc(dprime: float, n: int = 2001):
    """(fpr, tpr) grid of the equal-variance binormal ROC."""
    f = np.linspace(0.0, 1.0, n)
    t = np.empty_like(f)
    t[0] = 0.0
    t[1:] = norm.cdf(norm.ppf(np.clip(f[1:], 1e-15, 1 - 1e-15)) + dprime)
    return f, np.maximum.accumulate(t)


def empirical_roc(series: str = "macro", path: str | None = None):
    """(fpr, tpr, auc01) of a curve in results/germline_roc.csv, as a monotone step."""
    roc = pd.read_csv(path or os.path.join(RESULTS, "germline_roc.csv"))
    s = roc[roc.series == series].sort_values("fpr")
    if s.empty:
        raise ValueError("no such series: %r" % series)
    f = np.array(s.fpr, dtype=float)          # np.array, not .to_numpy(): a pandas view is
    t = np.array(s.tpr, dtype=float)          # read-only and the endpoint fix below fails
    f[0], t[0] = 0.0, 0.0
    return f, np.maximum.accumulate(t), float(s.auc01.iloc[0])


def task_a(f, t, prevalence: float, k: int, n_rep: int = N_REP, tpr_fn=None) -> dict:
    """Screen a repertoire of `n_rep`, take the top `k` by score.

    Solves for the operating point at which exactly `k` receptors are selected, then reports
    what that selection contains. Enrichment is precision/prevalence: the factor by which
    screening beats picking at random.

    `tpr_fn` matters. Taking the top 100 of 10^6 puts the operating point at FPR ~1e-4, and
    interpolating a ROC sampled on a uniform grid collapses TPR there -- a grid of 2001
    points has spacing 5e-4, so the whole high-specificity region falls inside the first
    interval and linear interpolation understates TPR several-fold. Pass the closed-form
    TPR(FPR) whenever one exists (it does for the binormal); the grid is then used only for
    an empirical curve, where the resolution limit is a property of the data rather than an
    artefact of the integration.
    """
    npos = n_rep * prevalence
    tpr_at = tpr_fn if tpr_fn is not None else (lambda x: float(np.interp(x, f, t)))
    fpr = float(brentq(lambda x: npos * tpr_at(x) + (n_rep - npos) * x - k,
                       1e-14, 1 - 1e-14))
    tpr = tpr_at(fpr)
    tp, fp = npos * tpr, (n_rep - npos) * fpr
    prec = tp / max(tp + fp, 1e-12)
    return dict(prevalence=prevalence, top_k=k, exp_tp=tp, exp_fp=fp, precision=prec,
                recall=tpr, enrichment=prec / prevalence)


def task_b(f, t, panel: int, k: int) -> float:
    """P(the cognate epitope ranks in the top `k` of a panel of `panel`).

    For a random cognate pair let U be the fraction of non-cognate pairs scoring above it;
    U has density dTPR along the ROC. Given U, the number of non-cognates above the cognate
    is Binom(panel-1, U), so the cognate is in the top k when that count is at most k-1.
    Chance is k/panel, attained when the ROC is the diagonal.
    """
    dt = np.diff(t)
    u = 0.5 * (f[1:] + f[:-1])
    p = binom.cdf(k - 1, panel - 1, np.clip(u, 0.0, 1.0))
    return float(np.sum(p * dt) / max(dt.sum(), 1e-12))


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 100:
        return "%.0f" % v
    if a >= 10:
        return "%.1f" % v
    if a >= 1:
        return "%.2f" % v
    return "%.3f" % v


def main():
    curves = {}
    for name, target in (("best", AUC_BEST), ("strong", STRONG)):
        dp = calibrate(target)
        f, t = binormal_roc(dp)
        # closed-form TPR for the screening task, whose operating point sits at FPR ~1e-4
        tpr_fn = (lambda x, _d=dp: float(norm.cdf(norm.ppf(min(max(x, 1e-300), 1 - 1e-16)) + _d)))
        curves[name] = dict(label="binormal at AUC0.1=%.2f" % target, f=f, t=t, tpr_fn=tpr_fn,
                            auc01=target, dprime=dp, full=float(norm.cdf(dp / np.sqrt(2))))
    f, t, a01 = empirical_roc("macro")
    curves["germline"] = dict(label="germline empirical (macro over 20 peptides)",
                              f=f, t=t, tpr_fn=None, auc01=a01, dprime=calibrate(a01),
                              full=float(norm.cdf(calibrate(a01) / np.sqrt(2))))

    rows_a, rows_b = [], []
    for key, c in curves.items():
        print("\n=== %s -- AUC0.1 %.4f, d'=%.3f, full AUC %.4f ==="
              % (c["label"], c["auc01"], c["dprime"], c["full"]))
        print("  %10s %6s %8s %10s %10s %8s %9s"
              % ("prevalence", "top k", "exp TP", "exp FP", "precision", "recall", "enrich"))
        for prev in PREVALENCES:
            for k in TOPK:
                r = task_a(c["f"], c["t"], prev, k, tpr_fn=c.get("tpr_fn"))
                rows_a.append(dict(curve=key, **r))
                print("  %10.0e %6d %8.3f %10.1f %10.5f %8.3f %8.1fx"
                      % (prev, k, r["exp_tp"], r["exp_fp"], r["precision"], r["recall"],
                         r["enrichment"]))
        print("  %7s %9s %8s %9s %10s" % ("panel M", "P(top1)", "chance", "P(top5)", "P(top10)"))
        for M in PANELS:
            p1, p5, p10 = (task_b(c["f"], c["t"], M, j) for j in (1, 5, 10))
            rows_b.append(dict(curve=key, panel=M, p_top1=p1, p_top5=p5, p_top10=p10,
                               chance_top1=1.0 / M))
            print("  %7d %9.4f %8.4f %9.4f %10.4f" % (M, p1, 1.0 / M, p5, p10))

    A = pd.DataFrame(rows_a); B = pd.DataFrame(rows_b)
    A.to_csv(os.path.join(RESULTS, "utility_screen.csv"), index=False)
    B.to_csv(os.path.join(RESULTS, "utility_panel.csv"), index=False)

    def a_at(curve, prev, k, col):
        return A[(A.curve == curve) & (A.prevalence == prev) & (A.top_k == k)][col].iloc[0]

    def b_at(curve, M, col):
        return B[(B.curve == curve) & (B.panel == M)][col].iloc[0]

    macros = {
        "utilNrep": "10^{6}",
        "utilPrev": "10^{-5}",
        "utilTopK": "100",
        # the benchmark's own reported score
        "utilTpBest": _fmt(a_at("best", 1e-5, 100, "exp_tp")),
        "utilPrecBest": "%.1e" % a_at("best", 1e-5, 100, "precision"),
        "utilEnrichBest": _fmt(a_at("best", 1e-5, 100, "enrichment")),
        "utilRecallBestHiK": _fmt(a_at("best", 1e-5, 10000, "recall")),
        "utilTpBestHiK": _fmt(a_at("best", 1e-5, 10000, "exp_tp")),
        # the same at a hypothetical strong predictor -- the range argument
        "utilTpStrong": _fmt(a_at("strong", 1e-5, 100, "exp_tp")),
        "utilPrecStrongPct": _fmt(100 * a_at("strong", 1e-5, 100, "precision")),
        # the germline classifier's empirical curve
        "utilAucGerm": "%.2f" % curves["germline"]["auc01"],
        "utilTpGerm": _fmt(a_at("germline", 1e-5, 100, "exp_tp")),
        "utilEnrichGerm": _fmt(a_at("germline", 1e-5, 100, "enrichment")),
        # panel identification
        "utilPanelBest": _fmt(b_at("best", 20, "p_top1")),
        "utilPanelGerm": _fmt(b_at("germline", 20, "p_top1")),
        "utilPanelChance": _fmt(b_at("best", 20, "chance_top1")),
        "utilPanelBestHundred": _fmt(b_at("best", 100, "p_top1")),
        "utilPanelStrong": _fmt(b_at("strong", 20, "p_top1")),
        "utilDprimeBest": "%.2f" % curves["best"]["dprime"],
        "utilFullBest": "%.2f" % curves["best"]["full"],
    }
    with open(os.path.join(ADAT, "utility_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "utility_screen.csv"))
    print("wrote %s" % os.path.join(RESULTS, "utility_panel.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "utility_macros.tex"), len(macros)))


def demo():
    """Self-check: the calibration inverts, and both tasks reduce to chance on a diagonal ROC."""
    dp = calibrate(AUC_BEST)
    assert abs(std_pauc01(dp) - AUC_BEST) < 1e-6, dp
    # a diagonal ROC is chance: enrichment 1x, and the cognate ranks top-k with probability k/M
    f = np.linspace(0, 1, 4001)
    chance = task_a(f, f, 1e-5, 1000)
    assert abs(chance["enrichment"] - 1.0) < 0.02, chance
    for M, k in ((20, 1), (100, 5)):
        assert abs(task_b(f, f, M, k) - k / M) < 0.01, (M, k, task_b(f, f, M, k))
    # a near-perfect ROC recovers almost every positive
    dp9 = calibrate(0.99)
    f9, t9 = binormal_roc(dp9)
    assert task_b(f9, t9, 20, 1) > 0.9, task_b(f9, t9, 20, 1)
    # monotone in the score: a better predictor is never worse on either task
    fb, tb = binormal_roc(calibrate(AUC_BEST))
    fs, ts = binormal_roc(calibrate(STRONG))
    assert task_a(fs, ts, 1e-5, 100)["exp_tp"] > task_a(fb, tb, 1e-5, 100)["exp_tp"]
    assert task_b(fs, ts, 20, 1) > task_b(fb, tb, 20, 1)
    print("utility.demo OK  (d'=%.3f at AUC0.1=%.2f; diagonal ROC gives %.2fx enrichment "
          "and P(top1 of 20)=%.3f)" % (dp, AUC_BEST, chance["enrichment"], task_b(f, f, 20, 1)))


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
