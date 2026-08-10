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
    # grand median of the three aggregates (in log space -> use the group means)
    med = float(np.median([high.mean(), floor.mean(), mid.mean()]))  # = mid.mean() ~1.2
    # high's tight CI excludes the median; a cohort centred on the median does not
    assert star_ci(high.mean() - 0.1, high.mean() + 0.1, med) == "*"
    assert star_ci(med - 0.2, med + 0.2, med) == ""
    assert fmt_p(1e-9) == "<0.001" and fmt_p(0.03).startswith("=")
    print("audit_stats.demo OK  (ANOVA p=%.1e, high-vs-floor Tukey p=%.1e)"
          % (a["p"], tk[("high", "floor")]))


if __name__ == "__main__":
    demo()
