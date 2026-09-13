# Significance annotations for the homology (Fig 1) and pairing (Fig 2) panels.
# 2026-07-12
#
# Three complementary layers, each honest about the small number of epitopes per
# anchor cohort (tcrvdb ~2, vdjdb_hq ~5, immrep22 ~7 vs immrep25/airr ~20):
#   * one-way ANOVA across the displayed cohorts on the per-epitope values
#     (homology on log10 S/N -- matching the geometric-mean design; pairing on
#     excess bits) -- a well-powered omnibus test that the cohorts differ at all;
#   * Tukey HSD post-hoc for the decisive IMMREP25-vs-{IMMREP22, MLR, AIRR
#     non-random} contrasts;
#   * a grand-median reference line (median of the displayed cohorts' aggregate
#     values) with a per-cohort star when that cohort's 95% CI excludes the line
#     -- consistent with the error bars already drawn, so it never manufactures
#     power the few-epitope anchors do not have.
from __future__ import annotations
import math
import numpy as np
from scipy import stats


def fmt_p(p) -> str:
    """Format a p-value for inline math, carrying its own relational operator.

    Returns e.g. '<0.001' or '=0.03' so prose can write ``$p\\macro$``.
    """
    if p is None or not np.isfinite(p):
        return "=\\mathrm{n/a}"
    if p < 1e-3:
        return "<0.001"
    if p < 1e-2:
        return "=%.3f" % p
    return "=%.2f" % p


def _clean(groups):
    out = []
    for g in groups:
        a = np.asarray(g, dtype=float)
        a = a[np.isfinite(a)]
        out.append(a)
    return out


def anova(groups) -> dict:
    """One-way ANOVA over a list of 1-D arrays. Returns F, df1, df2, p."""
    gs = [g for g in _clean(groups) if len(g) >= 2]
    k = len(gs)
    n = int(sum(len(g) for g in gs))
    if k < 2:
        return dict(F=np.nan, df1=np.nan, df2=np.nan, p=np.nan)
    F, p = stats.f_oneway(*gs)
    return dict(F=float(F), df1=k - 1, df2=n - k, p=float(p))


def kruskal(groups) -> dict:
    """Kruskal-Wallis omnibus (rank-based robustness cross-check)."""
    gs = [g for g in _clean(groups) if len(g) >= 1]
    if len(gs) < 2:
        return dict(H=np.nan, p=np.nan)
    H, p = stats.kruskal(*gs)
    return dict(H=float(H), p=float(p))


def tukey(groups, names) -> dict:
    """Tukey HSD across groups; returns {(name_i, name_j): p} for pairs with data."""
    arrs = _clean(groups)
    keep = [i for i, a in enumerate(arrs) if len(a) >= 2]
    if len(keep) < 2:
        return {}
    res = stats.tukey_hsd(*[arrs[i] for i in keep])
    out = {}
    for ii, i in enumerate(keep):
        for jj, j in enumerate(keep):
            if i < j:
                out[(names[i], names[j])] = float(res.pvalue[ii, jj])
    return out


def star_vs(tk: dict, cohort: str, ref: str, alpha: float = 0.05) -> str:
    """'*' if `cohort` differs from the reference cohort by Tukey HSD at `alpha`.

    Significance is defined against the set under audit (IMMREP25), not against a grand
    median of whichever cohorts a panel happens to show -- the latter moves when a bar is
    added or dropped, so it is not a property of the data.
    """
    if cohort == ref:
        return ""
    p = tk.get((cohort, ref), tk.get((ref, cohort)))
    if p is None or not np.isfinite(p):
        return ""
    return "*" if p < alpha else ""


# --------------------------------------------------------------------------- #
# Heteroscedastic and equivalence layer (round 3).
#
# The cohorts differ in between-epitope variance by orders of magnitude -- airr_control
# on TCRbeta has sd(log10 S/N) = 0.000 (every epitope pinned at S/N=1 by the pseudocount)
# against 0.710 for immrep22_true -- so the equal-variance one-way ANOVA above is not
# valid on its own and Welch's F is reported beside it. For "at background" contrasts we
# report the largest effect the data rule out rather than a non-significant p; on a ratio
# statistic aggregated as a geometric mean that bound must be built on log10 and read back
# as a FOLD-RATIO, never as an additive difference.
# --------------------------------------------------------------------------- #

def welch_anova(groups) -> dict:
    """Welch's heteroscedastic one-way F. Returns F, df1, df2, p."""
    gs = [g for g in _clean(groups) if len(g) >= 2]
    k = len(gs)
    if k < 2:
        return dict(F=np.nan, df1=np.nan, df2=np.nan, p=np.nan)
    n = np.array([len(g) for g in gs], dtype=float)
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    w = n / v
    # a zero-variance group carries infinite weight; fall back to the equal-variance F
    if not np.all(np.isfinite(w)):
        return anova(gs)
    mw = float((w * m).sum() / w.sum())
    num = float((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = float((((1 - w / w.sum()) ** 2) / (n - 1)).sum())
    den = 1.0 + 2.0 * (k - 2) / (k ** 2 - 1) * lam
    F = num / den
    df2 = (k ** 2 - 1) / (3.0 * lam)
    return dict(F=F, df1=k - 1, df2=float(df2), p=float(stats.f.sf(F, k - 1, df2)))


def games_howell(groups, names) -> dict:
    """Games-Howell post-hoc (Welch df + studentized range): {(name_i, name_j): p}.

    The unequal-variance counterpart of `tukey`, for the same reason `welch_anova` exists.
    """
    arrs = _clean(groups)
    keep = [i for i, a in enumerate(arrs) if len(a) >= 2 and a.var(ddof=1) > 0]
    k = len(keep)
    out = {}
    if k < 2:
        return out
    for ii in range(k):
        for jj in range(ii + 1, k):
            i, j = keep[ii], keep[jj]
            a, b = arrs[i], arrs[j]
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se = math.sqrt(va + vb)
            q = abs(a.mean() - b.mean()) / (se / math.sqrt(2.0))
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            out[(names[i], names[j])] = float(stats.studentized_range.sf(q, k, df))
    return out


def welch_log_bound(x, y, conf: float = 0.90, log: bool = True) -> dict:
    """Welch two-sample contrast with an equivalence bound, on the log10 scale.

    `x`, `y` are per-epitope values of a ratio statistic (S/N). With log=True the contrast
    is mean(log10 x) - mean(log10 y) and the interval is back-transformed, so `fold_lo` /
    `fold_hi` bracket the FOLD-RATIO x/y. `conf` is two-sided; 0.90 corresponds to TOST at
    alpha=0.05 per side, which is the correct width for an equivalence claim.
    """
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if log:
        a = np.log10(a[a > 0]); b = np.log10(b[b > 0])
    if len(a) < 2 or len(b) < 2:
        return dict(d=np.nan, se=np.nan, lo=np.nan, hi=np.nan, fold_lo=np.nan,
                    fold_hi=np.nan, bound=np.nan, p=np.nan, df=np.nan, n_x=len(a), n_y=len(b))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.nan
    d = float(a.mean() - b.mean())
    t = stats.t.ppf(0.5 + conf / 2.0, df) if np.isfinite(df) else np.nan
    lo, hi = d - t * se, d + t * se
    p = float(2 * stats.t.sf(abs(d / se), df)) if se > 0 else np.nan
    return dict(d=d, se=float(se), lo=float(lo), hi=float(hi),
                fold_lo=float(10 ** lo) if log else float(lo),
                fold_hi=float(10 ** hi) if log else float(hi),
                bound=float(max(abs(lo), abs(hi))), p=p, df=float(df),
                n_x=int(len(a)), n_y=int(len(b)))


def geo_ci(x, null: float = 1.0, conf: float = 0.95) -> dict:
    """Geometric mean of a ratio statistic with its CI, and a test against `null`.

    Matches `homology.dataset_sn`'s aggregation (mean of log10 with a log10 SE across
    epitopes), and adds the two-sided p against log10(null) so "excludes background" is a
    stated test rather than an eyeballed interval.
    """
    a = np.asarray(x, dtype=float); a = a[np.isfinite(a) & (a > 0)]
    if len(a) < 2:
        return dict(gm=float(10 ** np.log10(a).mean()) if len(a) else np.nan,
                    lo=np.nan, hi=np.nan, p=np.nan, n=len(a))
    l = np.log10(a)
    se = l.std(ddof=1) / math.sqrt(len(l))
    t = stats.t.ppf(0.5 + conf / 2.0, len(l) - 1)
    p = float(2 * stats.t.sf(abs((l.mean() - math.log10(null)) / se), len(l) - 1)) if se > 0 else np.nan
    return dict(gm=float(10 ** l.mean()), lo=float(10 ** (l.mean() - t * se)),
                hi=float(10 ** (l.mean() + t * se)), p=p, n=int(len(l)),
                sd_log=float(l.std(ddof=1)))


def poisson_rate_upper(k: int, exposure: float, conf: float = 0.975) -> float:
    """Exact one-sided upper bound on a Poisson rate: the honest statistic when a count is 0.

    With zero within-epitope neighbour pairs the pseudocount pins S/N at exactly 1 and its
    interval collapses to zero width, so no ratio may be quoted; this bounds the rate itself
    (k events in `exposure` pair-exposures) without any pseudocount or asymptotics.
    """
    if exposure <= 0:
        return np.nan
    return float(stats.chi2.ppf(conf, 2 * (k + 1)) / 2.0 / exposure)


def demo():
    # A clearly-separated cohort (high), a floor cohort (~1), and a mid cohort.
    rng = np.random.default_rng(0)
    high = rng.normal(3.0, 0.2, 12)          # log10 S/N ~ 1000
    floor = rng.normal(0.0, 0.05, 20)        # log10 S/N ~ 1
    mid = rng.normal(1.2, 0.3, 20)           # log10 S/N ~ 16
    a = anova([high, floor, mid])
    assert a["p"] < 1e-3 and a["df1"] == 2, a
    tk = tukey([high, floor, mid], ["high", "floor", "mid"])
    assert tk[("high", "floor")] < 1e-3, tk
    assert fmt_p(1e-9) == "<0.001" and fmt_p(0.03).startswith("=")

    # Welch agrees with the omnibus when variances differ several-fold
    wa = welch_anova([high, floor, mid])
    assert wa["p"] < 1e-3 and wa["df1"] == 2, wa
    gh = games_howell([high, floor, mid], ["high", "floor", "mid"])
    assert gh[("high", "floor")] < 0.05, gh

    # equivalence bound on a fold-ratio: two cohorts an order of magnitude apart, and two
    # drawn from the same distribution (where the bound must be the informative output)
    sep = welch_log_bound(10 ** high, 10 ** floor)
    assert sep["fold_lo"] > 100 and sep["p"] < 1e-3, sep
    same = welch_log_bound(10 ** mid, 10 ** rng.normal(1.2, 0.3, 20))
    assert same["p"] > 0.05 and same["fold_lo"] < 1 < same["fold_hi"], same

    # geometric-mean CI reproduces the aggregation used for S/N, and excludes background
    g = geo_ci(10 ** mid)
    assert g["lo"] > 1 and g["p"] < 1e-3 and g["n"] == 20, g
    gf = geo_ci(10 ** floor)
    assert gf["lo"] < 1 < gf["hi"], gf

    # an exact rate bound for a zero count, where a ratio is undefined
    ub = poisson_rate_upper(0, 420)
    assert abs(ub - 3.689 / 420) < 1e-4, ub
    assert poisson_rate_upper(4, 420) > ub

    print("audit_stats.demo OK  (ANOVA p=%.1e, Welch p=%.1e, high-vs-floor Tukey p=%.1e, "
          "fold-ratio 90%% CI %.0f-%.0fx, zero-count rate bound %.2e)"
          % (a["p"], wa["p"], tk[("high", "floor")], sep["fold_lo"], sep["fold_hi"], ub))


if __name__ == "__main__":
    demo()
