#!/usr/bin/env python
# 2026-09-14  What IMMREP25's negative design can register at all: the generic-binder channel
# is zeroed by construction, at every label-noise level.
#
# IMMREP25 builds each peptide's negatives by re-pairing its receptors with the other nine
# peptides OF THE SAME ALLELE. So within an allele there is one fixed set of 500 receptors, and
# every one of them appears ONCE as a labelled positive (for its cognate peptide) and NINE TIMES
# as a labelled negative (for the others). Mislabelling is therefore a property of the RECEPTOR,
# not of the (receptor, peptide) row: a record that is not a genuine binder is wrong once in the
# positive class and nine times in the negative class, of the same allele.
#
# Consider the oracle that answers the generic question -- "does this receptor bind anything?" --
# perfectly, and is given no peptide. For the scored peptide p, its 50 labelled positives contain
# a fraction (1-f) of genuine binders, and so do its 450 labelled negatives, because those
# negatives ARE the other nine peptides' labelled positives and carry the same false-positive
# rate. The two class-conditional score distributions are identical, the ROC is the diagonal, and
# the McClish-standardised partial AUC is exactly 1/2.
#
# This holds at EVERY f, including f = 0: even with perfect labels, a perfect generic
# binder/non-binder discriminator scores exactly chance on this benchmark. The design does not
# merely make the generic channel hard to exploit, it removes it. Whatever a predicted structure,
# a docking score or an interface-confidence statistic contributes, it cannot be contributed
# through that channel -- which is the channel such statistics measure (see iptm_predictor).
#
# The contrast is the point, and the self-check is built on it: the oracle that DOES know the
# scored peptide -- s(r) = 1{r binds p} -- reproduces the independently derived label-noise
# ceiling 1 - f/2 exactly, per peptide, from this same machinery. Two modules, two derivations,
# one number: validation_efficiency.auc_ceiling(f) is recovered here to floating-point.
#
# The grid is defined in COUNTS, not in decimals, because on a pool of m receptors the realised
# false-positive fraction is quantised to multiples of 1/m -- there is no such benchmark as one
# with f = 0.25 on 50 positives. Writing the grid as f = k/m makes the comparison against the
# analytic ceiling exact instead of approximate; an earlier decimal grid silently rounded 12.5
# to 12 and compared f = 0.24 against the ceiling at 0.25, which this module's own check caught.
#
# Nothing here is simulated except the finite-sample envelope. The equalities are exact and are
# asserted, not estimated; the random-allocation draws exist only to report how far a realised
# benchmark of this size wanders from the analytic value.
#
# Run: python src/negative_design.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

import blind_ceiling as B                                          # noqa: E402
import epitope_free as E                                           # noqa: E402
import validation_efficiency as V                                  # noqa: E402

MAX_FPR = 0.1
AUC_BEST = 0.60          # best of 126 IMMREP25 submissions (Richardson 2026)
K_STEP = 2               # grid stride in false positives per pool; f = k/m is then exact
N_DRAW = 200             # draws for the finite-sample envelope only; the equalities are exact
SEED = 0


def pools() -> list[dict]:
    """The benchmark's own per-allele pool structure, read rather than assumed."""
    b = pd.read_csv(BENCH, sep="\t")
    rec = b[b.label == 1][["peptide", "hla"]].reset_index(drop=True)
    assert len(rec) == 1000, len(rec)
    out = []
    for hla, g in rec.groupby("hla"):
        pool = g.peptide.to_numpy()
        peps = sorted(g.peptide.unique())
        m = len(pool) // len(peps)
        assert m * len(peps) == len(pool), "pools are not equal-sized; the algebra assumes they are"
        out.append(dict(hla=hla, pool=pool, peps=peps, m=m))
    return out


def binders_exact(pool: np.ndarray, peps, k: int) -> np.ndarray:
    """Mark exactly `k` receptors per pool as non-binders, so the realised fraction is k/m.

    Which receptors are marked is irrelevant: the oracle's score is binary, so only the per-pool
    counts enter the ROC.
    """
    binder = np.ones(len(pool), dtype=float)
    for p in peps:
        idx = np.flatnonzero(pool == p)
        assert k <= len(idx), "k=%d exceeds the pool size %d" % (k, len(idx))
        binder[idx[:k]] = 0.0
    return binder


def binders_random(pool: np.ndarray, f: float, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli mislabelling: f is the expectation, not the realised fraction."""
    return (rng.random(len(pool)) >= f).astype(float)


def generic_partial(binder: np.ndarray, pool: np.ndarray, peps) -> np.ndarray:
    """Per-peptide standardised partial AUC of the peptide-blind generic-binder oracle.

    One score per receptor, reused for all K per-peptide evaluations -- the oracle never sees
    which peptide is being scored.
    """
    return E.per_peptide_pauc(binder, pool, peps)


def specific_partial(binder: np.ndarray, pool: np.ndarray, peps) -> np.ndarray:
    """Same, for the oracle that knows the scored peptide: s(r) = 1{r genuinely binds p}.

    A different score vector per peptide, so this cannot go through per_peptide_pauc.
    """
    vals = []
    for p in peps:
        s = binder * (pool == p)
        order, starts = E.tie_groups(s)
        vals.append(E.pauc01((pool == p).astype(float), order, starts))
    return np.array(vals)


def generic_full(binder: np.ndarray, pool: np.ndarray, peps) -> np.ndarray:
    """Per-peptide FULL AUC of the generic oracle, for the sum_p AUC_p = K/2 identity."""
    return np.array([roc_auc_score((pool == p).astype(int), binder) for p in peps])


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for r in pools():
        pool, peps, m = r["pool"], r["peps"], r["m"]
        K = len(peps)
        for k in range(0, m, K_STEP):
            f = k / m                                    # exact by construction
            bx = binders_exact(pool, peps, k)
            gx = generic_partial(bx, pool, peps)
            sx = specific_partial(bx, pool, peps)
            fx = generic_full(bx, pool, peps)
            draws = [float(np.mean(generic_partial(binders_random(pool, f, rng), pool, peps)))
                     for _ in range(N_DRAW)]
            rows.append(dict(
                hla=r["hla"], n_pools=K, pool_size=m, n_false=k, f=f,
                generic_exact=float(np.mean(gx)),
                generic_exact_min=float(np.min(gx)), generic_exact_max=float(np.max(gx)),
                generic_full_sum=float(np.sum(fx)),
                specific_exact=float(np.mean(sx)),
                ceiling_analytic=V.auc_ceiling(f),
                rand_mean=float(np.mean(draws)),
                rand_lo=float(np.min(draws)), rand_hi=float(np.max(draws))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, "negative_design.csv"), index=False)

    K = int(t.n_pools.iloc[0])
    m = int(t.pool_size.iloc[0])
    dev_pep = float(np.max(np.abs(np.r_[t.generic_exact_min.to_numpy(),
                                        t.generic_exact_max.to_numpy()] - 0.5)))
    sum_dev = float(np.max(np.abs(t.generic_full_sum - K / 2)))
    ceil_dev = float(np.max(np.abs(t.specific_exact - t.ceiling_analytic)))
    lo, hi = float(t.rand_lo.min()), float(t.rand_hi.max())
    below = int((t.rand_lo < 0.5 - 1e-12).sum())
    blind = B.ceiling(K, m)
    f_lo, f_hi = float(t.f.min()), float(t.f.max())

    print("=== the generic-binder channel, over f = k/%d for k = 0..%d step %d (%d values, "
          "%d alleles) ===" % (m, int(t.n_false.max()), K_STEP, t.f.nunique(), t.hla.nunique()))
    print("peptide-blind generic oracle, macro-AUC_%.1f: exactly 0.5 at every f in [%.2f, %.2f] "
          "(max |deviation| over all %d peptide-by-f cells = %.3g)"
          % (MAX_FPR, f_lo, f_hi, len(t) * K, dev_pep))
    print("  finite-sample envelope, %d Bernoulli draws per cell: %.4f - %.4f" % (N_DRAW, lo, hi))
    print("  and it is ONE-SIDED: %d of %d cells put any draw below chance, because for equal "
          "pools sum_p (f_neg - f_pos) = 0 for any allocation, so only the second-order term "
          "survives" % (below, len(t)))
    print("  sum_p full AUC_p = K/2 = %.1f throughout (max |deviation| = %.3g)" % (K / 2, sum_dev))
    print("peptide-AWARE oracle reproduces validation_efficiency.auc_ceiling(f) exactly "
          "(max |deviation| = %.3g): %.4f at f=0, %.4f at f=0.5"
          % (ceil_dev, V.auc_ceiling(0.0), V.auc_ceiling(0.5)))
    print("so the leaderboard's %.2f sits %.4f above a channel this design zeroes, and %.4f above "
          "the peptide-blind oracle ceiling %.4f"
          % (AUC_BEST, AUC_BEST - 0.5, AUC_BEST - blind, blind))

    macros = {
        "ndKpools": "%d" % K,
        "ndPoolSize": "%d" % m,
        "ndFlo": "%.2f" % f_lo,
        "ndFhi": "%.2f" % f_hi,
        "ndNf": "%d" % t.f.nunique(),
        "ndNcells": "%d" % (len(t) * K),
        "ndNdraw": "%d" % N_DRAW,
        "ndGeneric": "0.5",
        "ndGenericDev": "<10^{-12}" if dev_pep < 1e-12 else "%.3g" % dev_pep,
        "ndRandLo": "%.4f" % lo,
        "ndRandHi": "%.4f" % hi,
        "ndRandBelow": "%d" % below,
        "ndFullSum": "%d" % (K // 2) if K % 2 == 0 else "%.1f" % (K / 2),
        "ndCeilDev": "<10^{-12}" if ceil_dev < 1e-12 else "%.3g" % ceil_dev,
        "ndSpecificAtZero": "%.4f" % V.auc_ceiling(0.0),
        "ndSpecificAtHalf": "%.4f" % V.auc_ceiling(0.5),
        "ndGenericGap": "%.4f" % (AUC_BEST - 0.5),
    }
    with open(os.path.join(ADAT, "negative_design_macros.tex"), "w") as fh:
        for kk, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (kk, v))
    print("\nwrote results/negative_design.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "negative_design_macros.tex"), len(macros)))


def demo():
    """Self-check on a synthetic 10x50 allele, no data read.

    The load-bearing assertion is the third: the peptide-AWARE oracle must reproduce
    validation_efficiency.auc_ceiling(f) to floating-point. That ties this module's ROC
    bookkeeping to a ceiling derived independently in another module, so an error in either
    shows up here rather than in the manuscript. It already did its job once, on the decimal
    grid described in the header.
    """
    K, m = 10, 50
    peps = list(range(K))
    pool = np.array([p for p in peps for _ in range(m)])

    for k in [0, 5, 12, 25, 40, 47]:
        f = k / m
        b = binders_exact(pool, peps, k)
        # 1. the generic oracle is at chance for EVERY peptide, not merely on average
        g = generic_partial(b, pool, peps)
        assert np.all(np.abs(g - 0.5) < 1e-12), (
            "generic oracle left chance at f=%g (max dev %.3g)" % (f, np.max(np.abs(g - 0.5))))
        # 2. and the full-AUC identity holds, as it must for any receptor-level score
        assert abs(np.sum(generic_full(b, pool, peps)) - K / 2) < 1e-9, (
            "sum_p AUC_p != K/2 at f=%g" % f)
        # 3. THE check: the peptide-aware oracle is the label-noise ceiling, from another module
        s = specific_partial(b, pool, peps)
        want = V.auc_ceiling(f)
        assert np.all(np.abs(s - want) < 1e-12), (
            "peptide-aware oracle %.12f != auc_ceiling(%g) = %.12f" % (s.mean(), f, want))
        # 4. the two oracles must differ: knowing the peptide is worth something at every f here
        assert s.mean() > g.mean() + 1e-9, "the two oracles coincide at f=%g" % f

    # 5. the pinning survives ANY realised allocation, not merely the balanced one. For equal
    #    pools f_neg(p) is the mean of the other K-1 pools' false fractions, so
    #    sum_p (f_neg(p) - f_pos(p)) = S - S = 0 identically -- the first-order term cancels for
    #    every realisation, and a draw can only leave chance through a one-signed second-order
    #    term. Hence the finite-sample envelope reported by main() is one-sided.
    rng0 = np.random.default_rng(7)
    for _ in range(25):
        b = binders_random(pool, 0.5, rng0)
        fp = np.array([1.0 - b[pool == p].mean() for p in peps])
        fn = np.array([1.0 - b[pool != p].mean() for p in peps])
        tot = float(np.sum(fn - fp))
        assert abs(tot) < 1e-12, "sum_p (f_neg - f_pos) != 0 for a realisation (%.3g)" % tot
        mac = float(np.mean(generic_partial(b, pool, peps)))
        assert mac >= 0.5 - 1e-9, "a realisation fell below chance (%.9f)" % mac

    # 6. the generic oracle is a receptor-level score, so it cannot clear the peptide-blind bound
    assert 0.5 <= B.ceiling(K, m) + 1e-12, "chance exceeds the peptide-blind ceiling"
    # 6. the exactness is not an artefact of the deterministic allocation: Bernoulli noise is
    #    unbiased about 0.5, so a large draw must sit close to it
    rng = np.random.default_rng(SEED)
    rand = np.mean([np.mean(generic_partial(binders_random(pool, 0.5, rng), pool, peps))
                    for _ in range(50)])
    assert abs(rand - 0.5) < 0.01, "random allocation is biased away from chance (%.4f)" % rand

    print("negative_design.demo OK  (generic oracle exactly 0.5 per peptide at f = 0 ... %.2f; "
          "sum_p AUC_p = %.1f; peptide-aware oracle reproduces auc_ceiling to 1e-12, %.4f -> %.4f "
          "over f = 0 -> 0.5; 50 Bernoulli draws at f=0.5 gave %.4f)"
          % (47 / m, K / 2, V.auc_ceiling(0.0), V.auc_ceiling(0.5), rand))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
