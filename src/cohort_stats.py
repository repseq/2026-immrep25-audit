#!/usr/bin/env python
# 2026-09-13  Would we have seen it? Detection probability at the benchmark's own geometry.
#
# "Failure to detect is not absence" is the reviewers' central objection, and it is answered
# by a measurement rather than an argument: rarefy every reference cohort to IMMREP25's exact
# design -- 20 epitopes of 50 receptors -- and count how often each one still separates from
# background. If a cohort's known convergence is detected in essentially every draw at this
# size, then the design resolves effects of that magnitude and IMMREP25's reading is
# informative. If it is not, we say so.
#
# The companion quantity is the minimum detectable effect. For a one-sample t on E epitopes
# with between-epitope spread sigma_log,
#
#     Delta_log10 (80% power, alpha=0.05 two-sided) = (t_{0.975,E-1} + t_{0.80,E-1}) * sigma_log / sqrt(E)
#
# reported as a FOLD-RATIO (10**Delta), never as an additive difference: S/N is a ratio
# aggregated as a geometric mean, so the only scale on which its uncertainty is symmetric is
# log10. This is the same aggregation homology.dataset_sn already uses.
#
# Three things this module is careful about:
#
#   * The resampling unit is the receptor nested in the epitope; the INFERENTIAL unit is the
#     epitope. Receptors within an epitope are not independent observations of convergence, so
#     the test is across epitopes, and rarefaction is without replacement -- with-replacement
#     draws manufacture convergence, because a duplicated receptor is simultaneously a
#     same-epitope and a zero-distance pair (panel1_q.py documents the same hazard for Q).
#
#   * You cannot rarefy UP. TCRvdb(+) has 2 qualifying epitopes and IMMREP22(+) has 7, so the
#     20-epitope geometry is unreachable for them and the power statement for those cohorts is
#     made at their own epitope count, reported per cohort. A single headline power number
#     across cohorts would be wrong, since the minimum detectable effect scales as 1/sqrt(E).
#
#   * log10 S/N is a mixture with an atom at zero: the pseudocount pins any epitope with no
#     within- and no cross-epitope pair at exactly S/N=1. A t-test on it is mis-specified in
#     proportion to how many such epitopes a draw contains, so every draw reports its pinned
#     count and the summary carries the mean.
#
# Run: python src/cohort_stats.py      (--demo for the self-check)
from __future__ import annotations
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

from audit_stats import anova, games_howell, tukey, welch_anova   # noqa: E402
from cohorts import build_cohorts, COHORT_META                    # noqa: E402
from homology import dataset_sn, per_epitope_sn                   # noqa: E402

TARGET_EP = 20        # IMMREP25's epitope count
# IMMREP25 nominally carries 50 receptors per peptide but its per-epitope counts are 48-50
# after de-duplication, so k=50 is degenerate: only 12 (TCRalpha) / 15 (TCRbeta) epitopes
# qualify at all, and "sample 50 of 50" returns the same subset every draw -- which made the
# reported detection rate one fixed draw repeated, not a resampling result. k=40 sits below
# the minimum, so every epitope qualifies and the draw is genuinely stochastic.
TARGET_K = 40
N_DRAWS = 200         # draws per cohort and chain; ~0.3 s each
POWER = 0.80
ALPHA = 0.05
SEED = 0
COHORTS = ("immrep25_pos", "vdjdb_hq", "vdjdb_lq", "immrep22_true", "tcrvdb_true",
           "pairseq_mock", "mlr_prolif")


def mde_fold(sigma_log: float, n_ep: int, power: float = POWER, alpha: float = ALPHA) -> float:
    """Smallest detectable geometric-mean S/N at `power`, as a fold-ratio above background."""
    if n_ep < 2 or not np.isfinite(sigma_log) or sigma_log <= 0:
        return np.nan
    t_a = stats.t.ppf(1 - alpha / 2, n_ep - 1)
    t_b = stats.t.ppf(power, n_ep - 1)
    return float(10 ** ((t_a + t_b) * sigma_log / math.sqrt(n_ep)))


def rarefy(long_chain: pd.DataFrame, n_ep: int, k: int, rng) -> pd.DataFrame | None:
    """Sample `n_ep` epitopes having >= k records, then exactly k receptors from each.

    Without replacement throughout. Returns None when the cohort cannot supply the geometry.
    """
    vc = long_chain.epitope.value_counts()
    eligible = vc[vc >= k].index.to_numpy()
    if eligible.size == 0:
        return None
    take = eligible if eligible.size <= n_ep else rng.choice(eligible, n_ep, replace=False)
    parts = []
    for ep in take:
        sub = long_chain[long_chain.epitope == ep]
        parts.append(sub.iloc[rng.choice(len(sub), k, replace=False)])
    return pd.concat(parts, ignore_index=True)


def draw_stats(sub: pd.DataFrame, k: int) -> dict | None:
    """One rarefied draw: geometric-mean S/N, its test against background, pinned count."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        pe = per_epitope_sn(sub, max_d=1, min_n=k, seed=0)
    if len(pe) < 2:
        return None
    v = pe.sn1.to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return None
    lg = np.log10(v)
    se = lg.std(ddof=1) / math.sqrt(len(lg))
    p = float(2 * stats.t.sf(abs(lg.mean() / se), len(lg) - 1)) if se > 0 else np.nan
    return dict(n_ep=len(v), sn=float(10 ** lg.mean()), sigma_log=float(lg.std(ddof=1)),
                p=p, detected=bool(np.isfinite(p) and p < ALPHA and lg.mean() > 0),
                n_pinned=int(np.sum(np.isclose(v, 1.0, atol=1e-9))))


def main():
    rng = np.random.default_rng(SEED)
    coh = build_cohorts()
    rows, per_draw = [], []
    for name in COHORTS:
        if name not in coh:
            continue
        for chain in ("A", "B"):
            lc = coh[name]["long"]
            lc = lc[lc.chain == chain]
            if lc.empty:
                continue
            got = [d for d in (draw_stats(s, TARGET_K) for s in
                               (rarefy(lc, TARGET_EP, TARGET_K, rng) for _ in range(N_DRAWS))
                               if s is not None) if d is not None]
            if not got:
                vc = lc.epitope.value_counts()
                print("  %-14s TCR%s: cannot reach %d receptors/epitope (max %d) -- skipped"
                      % (name, chain, TARGET_K, int(vc.max()) if len(vc) else 0))
                continue
            g = pd.DataFrame(got)
            g.insert(0, "chain", chain); g.insert(0, "cohort", name)
            per_draw.append(g)
            sig = float(np.median(g.sigma_log))
            e = int(np.median(g.n_ep))
            # A zero-variance S/N across draws means rarefaction returned the same subset
            # every time -- k at or above the cohort's per-epitope counts. The resulting
            # "detection rate" would be one draw repeated, so flag it rather than report it
            # as a resampling statistic.
            degenerate = bool(len(g) > 1 and np.isclose(g.sn.std(ddof=0), 0.0))
            if degenerate:
                print("  %-14s TCR%s: WARNING rarefaction is degenerate at k=%d (zero variance "
                      "over %d draws) -- detection rate is not a resampling statistic"
                      % (name, chain, TARGET_K, len(g)))
            rows.append(dict(cohort=name, chain=chain, n_draws=len(g), n_ep_median=e,
                             reached_target=bool(e >= TARGET_EP), degenerate=degenerate,
                             sn_median=float(np.median(g.sn)),
                             sn_lo=float(np.percentile(g.sn, 2.5)),
                             sn_hi=float(np.percentile(g.sn, 97.5)),
                             sigma_log=sig, detection_rate=float(g.detected.mean()),
                             mde_fold=mde_fold(sig, e),
                             pinned_mean=float(g.n_pinned.mean())))
    out = pd.DataFrame(rows)
    pd.concat(per_draw, ignore_index=True).to_csv(
        os.path.join(RESULTS, "cohort_power_draws.csv"), index=False)
    out.to_csv(os.path.join(RESULTS, "cohort_power.csv"), index=False)

    print("\n=== rarefied to %d epitopes x %d receptors, %d draws per cohort and chain ==="
          % (TARGET_EP, TARGET_K, N_DRAWS))
    print(out.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\ndetection_rate = fraction of draws whose 95%% CI on log10 S/N excludes background;")
    print("mde_fold = smallest geometric-mean S/N detectable at %d%% power, given that draw's"
          % int(100 * POWER))
    print("           between-epitope spread and epitope count (a fold-ratio, not a difference)")

    # cross-cohort tests on the FULL per-epitope values, equal- and unequal-variance side by side
    print("\n=== cross-cohort tests on log10 S/N (epitope as the unit) ===")
    tests = []
    for chain in ("A", "B"):
        groups, names = [], []
        for name in COHORTS:
            if name not in coh:
                continue
            lc = coh[name]["long"]; lc = lc[lc.chain == chain]
            if lc.empty:
                continue
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                pe = per_epitope_sn(lc, max_d=1, min_n=30, seed=1)
            v = pe.sn1.to_numpy(float) if len(pe) else np.array([])
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) >= 2:
                groups.append(np.log10(v)); names.append(name)
        if len(groups) < 2:
            continue
        a, w = anova(groups), welch_anova(groups)
        gh = games_howell(groups, names); tk = tukey(groups, names)
        sd = {n: float(g.std(ddof=1)) for n, g in zip(names, groups)}
        print("  TCR%s  equal-variance F=%.1f p=%.2e  |  Welch F=%.1f p=%.2e"
              % (chain, a["F"], a["p"], w["F"], w["p"]))
        print("        sd(log10 S/N) per cohort: %s"
              % ", ".join("%s %.3f" % (n, sd[n]) for n in names))
        ref = "immrep25_pos"
        for n in names:
            if n == ref:
                continue
            k1, k2 = (ref, n), (n, ref)
            print("        %-14s Tukey p=%-9.3g Games-Howell p=%-9.3g"
                  % (n, tk.get(k1, tk.get(k2, np.nan)), gh.get(k1, gh.get(k2, np.nan))))
            tests.append(dict(chain=chain, comparator=n, tukey_p=tk.get(k1, tk.get(k2, np.nan)),
                              games_howell_p=gh.get(k1, gh.get(k2, np.nan)),
                              sd_ref=sd[ref], sd_cmp=sd[n]))
        tests.append(dict(chain=chain, comparator="__omnibus__", anova_F=a["F"], anova_p=a["p"],
                          welch_F=w["F"], welch_p=w["p"]))
    pd.DataFrame(tests).to_csv(os.path.join(RESULTS, "cohort_tests.csv"), index=False)

    imm = out[out.cohort == "immrep25_pos"]
    hq = out[out.cohort == "vdjdb_hq"]
    macros = {"cpTargetEp": "%d" % TARGET_EP, "cpTargetK": "%d" % TARGET_K,
              "cpNdraws": "%d" % N_DRAWS, "cpPower": "%d" % int(100 * POWER)}
    for tag, sub in (("Imm", imm), ("Hq", hq)):
        for chain in ("A", "B"):
            r = sub[sub.chain == chain]
            if r.empty:
                continue
            macros["cp%sMde%s" % (tag, chain)] = "%.2f" % r.mde_fold.iloc[0]
            macros["cp%sDet%s" % (tag, chain)] = "%.0f" % (100 * r.detection_rate.iloc[0])
            macros["cp%sSn%s" % (tag, chain)] = "%.2f" % r.sn_median.iloc[0]
    with open(os.path.join(ADAT, "cohort_power_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "cohort_power.csv"))
    print("wrote %s" % os.path.join(RESULTS, "cohort_power_draws.csv"))
    print("wrote %s" % os.path.join(RESULTS, "cohort_tests.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "cohort_power_macros.tex"), len(macros)))


def demo():
    """Self-check: the power formula, rarefaction's guarantees, and Welch vs equal-variance."""
    # the minimum detectable effect must shrink with more epitopes and grow with more spread
    assert mde_fold(0.5, 20) < mde_fold(0.5, 7), "MDE must fall as epitopes are added"
    assert mde_fold(0.7, 20) > mde_fold(0.5, 20), "MDE must rise with between-epitope spread"
    # at sigma=0.510 and E=20 (VDJdb HQ's TCRbeta spread) the resolvable effect is ~2.2x
    got = mde_fold(0.510, 20)
    assert 2.0 < got < 2.4, got

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"epitope": np.repeat([f"E{i}" for i in range(5)], 60),
                       "cdr3": [f"CASS{i:04d}F" for i in range(300)],
                       "chain": "B", "v": "TRBV1", "j": "TRBJ1"})
    sub = rarefy(df, 3, 50, rng)
    assert sub is not None and len(sub) == 150 and sub.epitope.nunique() == 3, len(sub)
    # without replacement: no receptor may be drawn twice
    assert sub.cdr3.nunique() == len(sub), "rarefaction must sample without replacement"
    # a cohort that cannot supply k receptors per epitope must decline rather than invent them
    assert rarefy(df, 3, 500, rng) is None

    # Welch must not agree with the equal-variance F when variances differ by orders of
    # magnitude -- which is the situation across these cohorts
    tight = rng.normal(0.0, 0.001, 20)
    wide = rng.normal(0.3, 0.9, 20)
    a, w = anova([tight, wide]), welch_anova([tight, wide])
    assert np.isfinite(w["p"]) and w["p"] > a["p"], (a["p"], w["p"])
    print("cohort_stats.demo OK  (MDE %.2fx at sigma=0.51/E=20; rarefaction exact and without "
          "replacement; Welch p=%.3f vs equal-variance p=%.1e on unequal variances)"
          % (got, w["p"], a["p"]))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
