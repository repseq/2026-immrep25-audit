"""What a benchmark can measure when a fraction of its labelled positives are not binders.
# 2026-08-09  (rewritten: the score CEILING is the primary argument; see the note on f below)

Independent of any convergence probe. A pool-deconvolution assay calls a receptor positive because it
was enriched in a peptide pool, so a false positive is a receptor enriched for some other reason. Two
things follow, and both are properties of the LABELS rather than of any method:

1. Score ceiling. At test time a false positive is exchangeable with the negatives, so a fraction f
   of the positive class sits in the negative distribution however good the predictor is. For a
   predictor that ranks every genuine binder above every non-binder, the ROC over FPR<=r is the full
   box on (1-f) of the positives and the chance triangle on the remaining f, which after McClish
   standardisation gives, for ANY r and hence for the full AUC too,

       AUC_max(f) = 1 - f/2.

   Inverting it turns an observed score into a statement about the labels: a field whose best entry
   reaches A would be at its ceiling if f = 2(1-A). This is verified by simulation to three decimals
   (see the module self-check below).

2. Resolution. Two methods can be ordered only if their scores differ by more than the sampling noise
   of a per-peptide score, z*sigma. Label noise contracts every true difference by (1-f), so a
   clean-label difference must exceed z*sigma/(1-f) to survive. sigma is estimated for the statistic
   the challenge actually reports -- the McClish-standardised macro-AUC_0.1 -- at the benchmark's own
   geometry of N_POS positives against N_NEG same-MHC negatives per peptide, by simulation under a
   binormal model calibrated to the observed best score. The closed-form Hanley-McNeil expression is
   for the FULL AUC and does not apply to a partial one.

On f: no functional revalidation of IMMREP25 exists. The ~50% figure in the literature is TCRvdb's
revalidation of a small number of VDJdb epitopes (Messemaker et al. 2025), NOT a measurement on this
benchmark, and IMMREP25's pooled deconvolution has no reason to be cleaner. f is therefore treated
here as an unknown continuum over F_RANGE and never asserted for IMMREP25.

Usage: uv run python src/validation_efficiency.py   # writes .dat + macros
"""
from __future__ import annotations
import os

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")

AUC_BEST = 0.60       # best of the IMMREP25 submissions, macro-AUC0.1 (Richardson 2026)
N_SUB = 126           # submissions scored in the challenge (Richardson 2026)
MAX_FPR = 0.10        # the benchmark scores the low-false-positive region
N_POS = 50            # positive TCRs per IMMREP25 peptide (Richardson 2026 design)
N_NEG = 450           # same-MHC negatives per peptide: the other nine peptides' 50 receptors each
F_RANGE = (0.10, 0.50)  # plausible false-positive range for a pool-deconvolution assay
G_RANGE = (0.00, 0.25)  # cross-pool contamination: labelled negatives that are in fact binders.
                        # Forced above 0 by the within-MHC negative design, and likewise never
                        # asserted for IMMREP25 -- reported over a continuum, as f is.
Z = 1.96
N_SIM = 4000
SEED = 0


def auc_ceiling_full(f: float, g: float = 0.0) -> float:
    """Ceiling on the FULL AUC when a fraction f of the labelled positives are not binders and
    a fraction g of the labelled negatives are.

    For the ideal score s(r) = 1{r binds the scored peptide}, a positive/negative pair is
    ordered correctly with probability (1-f)(1-g) and tied with probability
    (1-f)g + f(1-g), so

        AUC_max(f, g) = (1-f)(1-g) + [(1-f)g + f(1-g)]/2 = 1 - f/2 - g/2.

    Symmetric and first-order in both, with the standing result as the g = 0 case.
    """
    return 1.0 - f / 2.0 - g / 2.0


def auc_ceiling(f: float, g: float = 0.0, max_fpr: float = MAX_FPR) -> float:
    """Ceiling on the McClish-standardised PARTIAL AUC over FPR<=max_fpr.

    This is NOT 1 - f/2 - g/2, and the difference is the whole point. Under the ideal score
    the mislabelled negatives -- genuine binders sitting in the negative class -- score at the
    TOP of the ranking, so the ROC is two straight segments,

        (0,0) -> (g, 1-f) -> (1,1),

    the first being the tied block of true binders (both the correctly labelled positives and
    the g of the negatives), the second the tied block of everything that does not bind. When
    g exceeds max_fpr the entire low-false-positive region the metric integrates over lies
    inside that first segment, and the attainable partial area collapses far below the
    full-AUC ceiling. At f=0, g=0.25 and max_fpr=0.1 the standardised partial ceiling is
    0.579 against a full-AUC ceiling of 0.875.

    So cross-pool contamination costs the reported metric much more than it costs the full
    AUC -- and IMMREP25's negatives are, by construction, the positives of the other nine
    peptides of the same MHC, so any receptor cross-reactive within an allele lands there.

    At g = 0 this reduces exactly to 1 - f/2, which is why the standing result is recovered:
    the raw area is (1-f)r + f r^2/2 and standardisation divides out r(1 - r/2).

    Correlated contamination is a separate channel and does not enter here. Clustering by
    donor, pool or well leaves both marginal fractions unchanged, so it does not move either
    ceiling; what it does is inflate the variance of a per-peptide score, raising the
    resolution floor z*sigma/(1-f). Ceiling and resolution must not be conflated.
    """
    r, p = max_fpr, 1.0 - f
    if g <= 0.0:
        area = p * r + (1.0 - p) * r * r / 2.0
    elif r <= g:
        area = (p / (2.0 * g)) * r * r                      # still on the first segment
    else:
        area = (p * g / 2.0                                  # all of the first segment
                + p * (r - g)                                # plus the second segment's base
                + ((1.0 - p) / (1.0 - g)) * (r - g) ** 2 / 2.0)
    lo, hi = 0.5 * r * r, r
    return 0.5 * (1.0 + (area - lo) / (hi - lo))


def _binormal_mu(target: float, max_fpr: float = MAX_FPR) -> float:
    """Unit-variance binormal separation whose standardised partial AUC equals `target`."""
    def pauc(mu):
        xs = np.linspace(1e-9, max_fpr, 20001)
        raw = np.trapezoid(norm.sf(norm.isf(xs) - mu), xs)
        return 0.5 * (1 + (raw - max_fpr ** 2 / 2) / (max_fpr - max_fpr ** 2 / 2))
    return float(brentq(lambda m: pauc(m) - target, 0.01, 6.0))


def sigma_per_peptide(mu: float, n_sim: int = N_SIM, seed: int = SEED) -> tuple[float, float]:
    """(mean, sd) of the per-peptide macro-AUC0.1 at the benchmark's geometry, by simulation."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(N_POS), np.zeros(N_NEG)]
    v = np.empty(n_sim)
    for i in range(n_sim):
        s = np.r_[rng.normal(mu, 1.0, N_POS), rng.normal(0.0, 1.0, N_NEG)]
        v[i] = roc_auc_score(y, s, max_fpr=MAX_FPR)
    return float(v.mean()), float(v.std(ddof=1))


def _selfcheck(seed: int = 1, n_sim: int = 1500) -> float:
    """Max |simulated - analytic| ceiling over the (f, g) grid.

    A perfectly separable signal must match 1 - f/2 - g/2: mislabelled positives are drawn
    from the negative distribution, and mislabelled negatives (cross-pool binders) from the
    positive one.
    """
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(N_POS), np.zeros(N_NEG)]
    err = 0.0
    for f in (0.0, 0.1, 0.3, 0.5, 0.8):
        for g in (0.0, 0.1, 0.25):
            v = np.empty(n_sim)
            w = np.empty(n_sim)
            for i in range(n_sim):
                k = rng.binomial(N_POS, f)          # positives that are not binders
                m = rng.binomial(N_NEG, g)          # negatives that are binders
                s = np.r_[rng.normal(50.0, 1.0, N_POS - k), rng.normal(0.0, 1.0, k),
                          rng.normal(50.0, 1.0, m), rng.normal(0.0, 1.0, N_NEG - m)]
                v[i] = roc_auc_score(y, s, max_fpr=MAX_FPR)
                w[i] = roc_auc_score(y, s)
            # the partial ceiling is the two-segment geometry; the full AUC is 1 - f/2 - g/2
            err = max(err, abs(v.mean() - auc_ceiling(f, g)),
                      abs(w.mean() - auc_ceiling_full(f, g)))
    return err


def run():
    mu = _binormal_mu(AUC_BEST)
    obs_mean, sigma = sigma_per_peptide(mu)
    crit = Z * sigma                        # smallest resolvable difference between two methods
    f_implied = 2.0 * (1.0 - AUC_BEST)      # f at which the best entry would BE the ceiling
    f_lo, f_hi = F_RANGE
    ceil_hi, ceil_lo = auc_ceiling(f_lo), auc_ceiling(f_hi)   # ceiling at the low/high end of f
    delta_crit = crit / (1.0 - f_hi)        # clean-label difference needed at the top of the range
    delta_obs_max = AUC_BEST - 0.5          # whole observed range of the field, above chance
    err = _selfcheck()
    assert err < 5e-3, "ceiling formula failed its self-check: max error %.4f" % err

    # curves for the figure: the ceiling, and the clean-label difference needed to stay resolvable
    fs = np.linspace(0.0, 0.9, 46)
    with open(os.path.join(ADAT, "valideff.dat"), "w") as fh:
        fh.write("# f auc_ceiling delta_needed\n")
        for f in fs:
            fh.write("%.3f %.4f %.4f\n" % (f, auc_ceiling(f), crit / max(1.0 - f, 1e-6)))

    # The structured-noise arm goes to its own file rather than a new column: valideff.dat's
    # schema is read by the existing gnuplot panel, and widening it would break that figure.
    g_lo, g_hi = G_RANGE
    with open(os.path.join(ADAT, "valideff_g.dat"), "w") as fh:
        fh.write("# f ceiling_g0 ceiling_g_mid ceiling_g_hi\n")
        for f in fs:
            fh.write("%.3f %.4f %.4f %.4f\n"
                     % (f, auc_ceiling(f, 0.0), auc_ceiling(f, g_hi / 2), auc_ceiling(f, g_hi)))

    macros = {
        "veAucBest": "%.2f" % AUC_BEST,
        "veNsub": "%d" % N_SUB,
        # the self-check the Methods quotes: max |simulated - (1 - f/2)| over the f grid
        "veSimErr": "%.3f" % err,
        "veNpos": "%d" % N_POS,
        "veNneg": "%d" % N_NEG,
        "veSigma": "%.3f" % sigma,
        "veCrit": "%.3f" % crit,
        "veFimplied": "%.2f" % f_implied,
        "veFpLo": "%d" % round(100 * f_lo),
        "veFpHi": "%d" % round(100 * f_hi),
        "veCeilLo": "%.2f" % ceil_lo,          # ceiling at f = F_RANGE[1]
        "veCeilHi": "%.2f" % ceil_hi,          # ceiling at f = F_RANGE[0]
        "veDeltaCrit": "%.2f" % delta_crit,
        "veDeltaObsMax": "%.2f" % delta_obs_max,
        # Structured noise: the benchmark's within-MHC negatives put genuine binders in the
        # negative class. On the FULL AUC that costs a symmetric g/2; on the partial AUC the
        # metric actually reports it costs far more, because those binders rank at the top and
        # so occupy the low-false-positive region the metric integrates over.
        "veGHi": "%d" % round(100 * g_hi),
        "veCeilFgHi": "%.2f" % auc_ceiling(f_hi, g_hi),
        "veCeilFgLo": "%.2f" % auc_ceiling(f_lo, g_hi),
        "veCeilGDrop": "%.2f" % (auc_ceiling(f_hi, 0.0) - auc_ceiling(f_hi, g_hi)),
        "veCeilGOnly": "%.2f" % auc_ceiling(0.0, g_hi),
        "veCeilGOnlyFull": "%.2f" % auc_ceiling_full(0.0, g_hi),
        "veCeilFullFgHi": "%.2f" % auc_ceiling_full(f_hi, g_hi),
    }
    with open(os.path.join(ADAT, "valideff_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("ceiling self-check: max |simulated - (1-f/2)| = %.4f" % err)
    print("per-peptide macro-AUC0.1 at %d vs %d: mean %.3f, sd %.4f  -> smallest resolvable "
          "difference z*sigma = %.3f" % (N_POS, N_NEG, obs_mean, sigma, crit))
    print("observed best %.2f equals the ceiling at f = %.2f" % (AUC_BEST, f_implied))
    print("at f = %.0f%%-%.0f%%, a PERFECT predictor would score %.2f-%.2f; the field's best is %.2f"
          % (100 * f_lo, 100 * f_hi, ceil_hi, ceil_lo, AUC_BEST))
    print("at f = %.0f%% a clean-label difference must exceed %.2f to be resolvable; the whole field "
          "spans %.2f above chance" % (100 * f_hi, delta_crit, delta_obs_max))
    print("wrote appendix/analysis/valideff.dat + valideff_macros.tex")


if __name__ == "__main__":
    run()
