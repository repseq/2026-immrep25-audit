#!/usr/bin/env python
# 2026-09-13  Does the cohort effect survive adjustment for the covariates it correlates with?
#
# The reviewers ask for donor, batch, pool, HLA, abundance, V/J usage and CDR3 length to be
# evaluated as predictors or covariates rather than left implicit. The question this answers
# is the one that matters for the ladder: once each epitope's own composition is accounted
# for, is there still a cohort effect, or was the ladder a proxy for how the cohorts differ in
# length, germline diversity and generation probability?
#
# Design. The unit is the EPITOPE, matching every other aggregation in the audit -- receptors
# within an epitope are not independent observations of convergence. The response is
# log10 S/N at d=1, and the model is
#
#     log10 S/N ~ cohort + CDR3 length + V entropy + J entropy + log Pgen + log degree + n
#
# fitted by OLS with HC3 heteroscedasticity-consistent standard errors. HC3 rather than a
# mixed model, and this is deliberate: donor, platform and assay are COHORT-LEVEL properties,
# nested completely inside the fixed effect of interest, so they have no within-cohort
# variation to estimate and a random intercept on them is not identifiable. Reporting one
# would invite exactly the methodological objection it was meant to answer. HC3 is the
# assumption-light alternative that keeps the estimand interpretable: the cohort coefficient
# is the adjusted difference in log10 S/N against the reference cohort.
#
# statsmodels is absent from this environment, so the fit is ~20 lines of numpy. That is also
# why the design matrix is rank-checked explicitly: a silent rank deficiency here would return
# a pseudo-inverse solution and an uninterpretable "adjusted" coefficient, and cohort-level
# covariates are precisely the way to create one.
#
# Abundance and donor are recorded as UNAVAILABLE rather than imputed. The IMMREP25 release
# carries no count field and no donor field (19 columns, verified in epitope_free.py), and for
# a cohort-level gap the missingness indicator is a linear combination of the cohort dummies --
# so including it breaks rank instead of absorbing anything. Row-level gaps are handled the
# other way, by placeholder plus indicator, which is what _pgen_cols does.
#
# Run: python src/covariates.py      (--demo for the self-check)
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
CACHE = os.path.join(REPO, "cache")

from cohorts import build_cohorts                     # noqa: E402
from homology import per_epitope_sn                   # noqa: E402

REF = "immrep25_pos"        # reference level, so coefficients read "against IMMREP25"
COHORTS = ("immrep25_pos", "vdjdb_hq", "vdjdb_lq", "immrep22_true", "tcrvdb_true",
           "pairseq_mock", "mlr_prolif")
MIN_N = 30
SEED = 0


def ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with HC3 robust standard errors. Returns (beta, se).

    HC3 scales each squared residual by 1/(1-h_i)^2 with h_i the leverage, which is the
    small-sample-safe member of the sandwich family and the right default when group sizes and
    variances differ as much as they do across these cohorts.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = (resid ** 2) / np.clip((1.0 - h) ** 2, 1e-12, None)
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _entropy(counts) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _pgen_lookup() -> dict:
    """cdr3 -> (log10 pgen, log10 1-mismatch degree), for the cohorts that have it cached."""
    out = {}
    for chain, tag in (("A", "A"), ("B", "B")):
        for name in ("immrep25_pos", "immrep22_true", "airr_control", "olga_random"):
            p = os.path.join(CACHE, "pgen1mm_%s_%s.tsv"
                             % (name.replace("_pos", "_pos").replace("immrep22_true",
                                                                     "immrep22_true"), tag))
            if not os.path.exists(p):
                continue
            t = pd.read_csv(p, sep="\t")
            for c, g, d in zip(t.cdr3, t.pgen, t.pgen1mm):
                if g > 0 and d > 0:
                    out[(chain, c)] = (float(np.log10(g)), float(np.log10(d)))
    return out


def epitope_table(coh, chain: str) -> pd.DataFrame:
    """One row per qualifying epitope: its S/N and its own composition."""
    pg = _pgen_lookup()
    rows = []
    for name in COHORTS:
        if name not in coh:
            continue
        lg = coh[name]["long"]
        lc = lg[lg.chain == chain]
        if lc.empty:
            continue
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            pe = per_epitope_sn(lc, max_d=1, min_n=MIN_N, seed=1)
        for r in pe.itertuples():
            sub = lc[lc.epitope == r.epitope]
            have = [pg[(chain, c)] for c in sub.cdr3 if (chain, c) in pg]
            rows.append(dict(
                cohort=name, chain=chain, epitope=r.epitope, sn=float(getattr(r, "sn1")),
                n=float(r.n),
                cdr3_len=float(sub.cdr3.str.len().mean()),
                v_entropy=_entropy(sub.v.value_counts().to_numpy()),
                j_entropy=_entropy(sub.j.value_counts().to_numpy()),
                log_pgen=float(np.mean([h[0] for h in have])) if have else np.nan,
                log_degree=float(np.mean([h[1] for h in have])) if have else np.nan))
    t = pd.DataFrame(rows)
    return t[np.isfinite(t.sn) & (t.sn > 0)].reset_index(drop=True)


def fit(t: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Adjusted cohort effects on log10 S/N, with HC3 SEs and an explicit rank check."""
    from scipy import stats
    t = t.copy()
    y = np.log10(t.sn.to_numpy(float))
    levels = [c for c in COHORTS if c in set(t.cohort) and c != REF]
    cols, names = [np.ones(len(t))], ["intercept"]
    for lv in levels:
        cols.append((t.cohort == lv).to_numpy(float)); names.append("cohort[%s]" % lv)
    used, miss_patterns = [], []
    for c in covariates:
        v = t[c].to_numpy(float)
        if not np.isfinite(v).any():
            continue
        # row-level gaps: placeholder plus an indicator, so no epitope is dropped
        miss = ~np.isfinite(v)
        v = np.where(miss, np.nanmedian(v), v)
        v = (v - v.mean()) / (v.std(ddof=0) or 1.0)
        cols.append(v); names.append(c); used.append(c)
        # One indicator per DISTINCT missingness pattern, not per covariate. log_pgen and
        # log_degree are read from the same cached lookup, so their patterns are identical and
        # adding one indicator each would put two copies of the same column in the design --
        # a rank deficiency that has nothing to do with the cohort dummies. (That is what the
        # guard caught: rank 12 of 13, with missingness at 23-61% within the VDJdb cohorts and
        # so demonstrably NOT cohort-level.)
        if miss.any() and 0 < miss.mean() < 1:
            key = miss.tobytes()
            if key not in miss_patterns:
                miss_patterns.append(key)
                cols.append(miss.astype(float))
                names.append("missing[%s]" % c)
    X = np.column_stack(cols)
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError("design matrix is rank-deficient (%d < %d): a cohort-level covariate "
                         "is collinear with the cohort dummies" % (rank, X.shape[1]))
    beta, se = ols_hc3(X, y)
    df = max(len(t) - X.shape[1], 1)
    p = 2 * stats.t.sf(np.abs(beta / np.clip(se, 1e-12, None)), df)
    return pd.DataFrame(dict(term=names, beta=beta, se=se, p=p,
                             fold=10 ** beta, n=len(t), k=X.shape[1], covariates=len(used)))


def main():
    coh = build_cohorts()
    COVS = ["cdr3_len", "v_entropy", "j_entropy", "log_pgen", "log_degree", "n"]
    out = []
    for chain in ("A", "B"):
        t = epitope_table(coh, chain)
        t.to_csv(os.path.join(RESULTS, "covariates_TR%s.csv" % chain), index=False)
        print("\n=== TCR%s: %d epitopes over %d cohorts ==="
              % (chain, len(t), t.cohort.nunique()))
        print("  covariate coverage: log_pgen %d/%d epitopes, log_degree %d/%d"
              % (int(np.isfinite(t.log_pgen).sum()), len(t),
                 int(np.isfinite(t.log_degree).sum()), len(t)))
        for label, covs in (("unadjusted", []), ("adjusted", COVS)):
            f = fit(t, covs).assign(chain=chain, model=label)
            out.append(f)
            print("  %s (k=%d):" % (label, f.k.iloc[0]))
            for r in f.itertuples():
                if r.term == "intercept":
                    continue
                star = "*" if r.p < 0.05 else " "
                print("    %-22s beta %+7.3f  se %.3f  fold %6.3f  p %8.2g %s"
                      % (r.term, r.beta, r.se, r.fold, r.p, star))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(RESULTS, "covariate_effects.csv"), index=False)

    def get(chain, model, term, col="beta"):
        m = res[(res.chain == chain) & (res.model == model) & (res.term == term)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    macros = {"cvMinN": "%d" % MIN_N,
              "cvNcov": "%d" % len(COVS),
              "cvCovList": "CDR3 length, V entropy, J entropy, $\\log P_\\mathrm{gen}$, "
                           "$\\log$ generation degree, epitope size"}
    for chain in ("A", "B"):
        for tag, term in (("Hq", "cohort[vdjdb_hq]"), ("Tt", "cohort[tcrvdb_true]")):
            for model, mtag in (("unadjusted", "Un"), ("adjusted", "Adj")):
                b = get(chain, model, term)
                if np.isfinite(b):
                    macros["cv%s%s%s" % (tag, mtag, chain)] = "%.2f" % (10 ** b)
                    macros["cv%s%sP%s" % (tag, mtag, chain)] = (
                        "<0.001" if get(chain, model, term, "p") < 1e-3
                        else "=%.3f" % get(chain, model, term, "p"))
    with open(os.path.join(ADAT, "covariate_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/covariates_TR{A,B}.csv, covariate_effects.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "covariate_macros.tex"), len(macros)))


def demo():
    """Self-check: HC3 against a known fit, and the rank guard against a cohort-level covariate."""
    rng = np.random.default_rng(SEED)
    # OLS point estimates must be exact on a noiseless linear system
    X = np.column_stack([np.ones(50), rng.normal(size=50), rng.normal(size=50)])
    truth = np.array([1.0, -2.0, 0.5])
    b, se = ols_hc3(X, X @ truth)
    assert np.allclose(b, truth, atol=1e-8), b
    assert np.allclose(se, 0.0, atol=1e-6), se
    # under heteroscedasticity HC3 must exceed the naive homoscedastic SE
    y = X @ truth + rng.normal(scale=np.abs(X[:, 1]) + 0.05)
    b2, se_hc3 = ols_hc3(X, y)
    XtX_inv = np.linalg.pinv(X.T @ X)
    resid = y - X @ b2
    s2 = resid @ resid / (len(y) - X.shape[1])
    se_ols = np.sqrt(np.diag(s2 * XtX_inv))
    assert se_hc3[1] > se_ols[1], (se_hc3[1], se_ols[1])

    # a covariate that is constant within cohort is collinear with the dummies: must raise,
    # not silently return a pseudo-inverse solution
    t = pd.DataFrame(dict(cohort=[REF] * 6 + ["vdjdb_hq"] * 6,
                          sn=np.r_[rng.uniform(1, 3, 6), rng.uniform(3, 9, 6)],
                          cdr3_len=[14.0] * 6 + [15.0] * 6,       # cohort-level by construction
                          v_entropy=rng.uniform(1, 4, 12), j_entropy=rng.uniform(1, 3, 12),
                          log_pgen=rng.uniform(-12, -8, 12), log_degree=rng.uniform(-10, -6, 12),
                          n=rng.uniform(30, 80, 12)))
    ok = fit(t, ["v_entropy"])
    assert "cohort[vdjdb_hq]" in set(ok.term)
    try:
        fit(t, ["cdr3_len"])
        raise AssertionError("rank guard failed to fire on a cohort-level covariate")
    except ValueError as exc:
        assert "rank-deficient" in str(exc), exc
    print("covariates.demo OK  (HC3 exact on a noiseless fit, wider than OLS under "
          "heteroscedasticity, and the rank guard fires on a cohort-level covariate)")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
