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
Z = 1.96
N_SIM = 4000
SEED = 0


def auc_ceiling(f: float) -> float:
    """Highest McClish-standardised AUC (any max_fpr) reachable when a fraction f of the labelled
    positives are drawn from the negative distribution. Also the full-AUC ceiling."""
    return 1.0 - f / 2.0


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
    """Max |simulated - analytic| ceiling over f; a perfectly separable signal must match 1 - f/2."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(N_POS), np.zeros(N_NEG)]
    err = 0.0
    for f in (0.0, 0.1, 0.3, 0.5, 0.8):
        v = np.empty(n_sim)
        for i in range(n_sim):
            k = rng.binomial(N_POS, f)
            s = np.r_[rng.normal(50.0, 1.0, N_POS - k), rng.normal(0.0, 1.0, k),
                      rng.normal(0.0, 1.0, N_NEG)]
            v[i] = roc_auc_score(y, s, max_fpr=MAX_FPR)
        err = max(err, abs(v.mean() - auc_ceiling(f)))
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
