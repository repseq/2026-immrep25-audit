#!/usr/bin/env python
# 2026-09-14  ipTM as a predictor: what interface confidence can and cannot discriminate, and
# what it is worth on IMMREP25's own metric.
#
# Two questions, one statistic.
#
# 1. WHICH KIND of non-binding does ipTM see? There are two, and they are not the same thing:
#      - BIOLOGICAL non-binding: a real, curated TCR-pMHC pairing that was tested and does not
#        reproduce. The complex genuinely cannot form. TCRvdb's negative class is exactly this --
#        former VDJdb positives that the MATCHMAKERS assay invalidated.
#      - COMBINATORIAL non-binding: a real TCR re-paired with a real pMHC it was never observed
#        with. Both halves are well-formed, so AlphaFold-Multimer folds them into a plausible
#        interface whether or not they bind.
#    IMMREP25's negatives are ENTIRELY combinatorial, by construction -- they are the other nine
#    same-MHC peptides' positives, re-paired. So if ipTM discriminates the first kind and not the
#    second, the benchmark contains only the class ipTM cannot see, and the ~0.55 plateau of the
#    structure-based entries has a cause rather than a mystery.
#
# 2. WHAT IS IT WORTH on the benchmark's own metric? We hold ipTM for IMMREP25's 1000 positives
#    but not for its 9000 negatives, so the benchmark's own task cannot be scored directly. We
#    score the positives against an EXTERNAL negative pool instead, which biases the answer
#    OPTIMISTIC: external negatives are true non-binders, whereas IMMREP25's are other epitopes'
#    genuine binders. An upper bound is the useful direction here.
#
# The thresholded score is where the algebra becomes exact. A binary score s = 1{ipTM > t} has an
# ROC with a single interior vertex (u, v) = (FPR, TPR), so the standardised partial AUC is a
# closed-form functional of (u, v) ALONE -- and that functional is already in the repo:
# validation_efficiency.auc_ceiling(f, g) is built on the same two-segment ROC (0,0) -> (g, 1-f)
# -> (1,1), so it equals our value at f = 1-v, g = u. Two consequences, both asserted below:
# the number does not depend on HOW MANY negatives the pool holds (only on the rate u), and the
# TPR required to reach any target score follows by inversion. That is what lets a pool of a
# different size than IMMREP25's stand in for it without special pleading.
#
# Per-epitope thresholds are reported as an ORACLE: choosing each peptide's own cut uses
# information no predictor is given. Named as such, like blind_ceiling's assignment step.
#
# Run: python src/iptm_predictor.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
HF = os.path.join(os.path.expanduser("~"), "hf", "tcren_structures")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
POS = os.path.join(RESULTS, "iptm_templates.csv")          # IMMREP25's 1000 folded positives

import epitope_free as E                                          # noqa: E402
import validation_efficiency as V                                 # noqa: E402

MAX_FPR = 0.1
AUC_BEST = 0.60          # best of 126 IMMREP25 submissions (Richardson 2026)
MIN_CLASS = 15           # per-epitope minimum per class, as in iptm_compare/plddt_panels
N_GRID = 400             # candidate thresholds, taken as quantiles of the observed ipTM

# The labelled reference sets, with what their negative class MEANS. Same schema throughout:
# y / epitope / mhc / iptm.
SETS = [
    ("TCRvdb", "biological", os.path.join(HF, "tcrvdb", "metadata.tsv")),
    ("VDJdb binder benchmark", "combinatorial",
     os.path.join(HF, "vdjdb_binder_benchmark", "metadata.tsv")),
    ("VDJdb free pool", "combinatorial", os.path.join(HF, "vdjdb_free_pool", "metadata.tsv")),
]
# The external negative pool for the IMMREP25 thresholding: IMMREP22 mismatched decoys, which are
# combinatorial re-pairings like IMMREP25's own negatives, and carry no positives at all.
NEG_POOL = os.path.join(HF, "immrep23_negatives", "immrep2022_negatives.tsv")


def std_pauc_binary(u: float, v: float, max_fpr: float = MAX_FPR) -> float:
    """Standardised partial AUC of a binary score whose single ROC vertex is (u, v) = (FPR, TPR).

    The two-segment ROC (0,0) -> (u,v) -> (1,1) is the one validation_efficiency.auc_ceiling
    already integrates, with g = u and 1-f = v. Reusing it keeps one derivation in one place.
    """
    return V.auc_ceiling(1.0 - v, u, max_fpr=max_fpr)


def required_tpr(u: float, target: float = AUC_BEST, max_fpr: float = MAX_FPR) -> float:
    """TPR a binary score must deliver at false-positive rate u to reach `target`."""
    lo = std_pauc_binary(u, 0.0, max_fpr)
    hi = std_pauc_binary(u, 1.0, max_fpr)
    if not (lo <= target <= hi):
        return float("nan")
    return float(brentq(lambda v: std_pauc_binary(u, v, max_fpr) - target, 0.0, 1.0))


def load_set(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t").dropna(subset=["iptm"])
    return d[["y", "epitope", "mhc", "iptm"]].rename(columns={"y": "label"})


def load_pool() -> pd.DataFrame:
    """The external negative pool. Different column names, all label 0."""
    d = pd.read_csv(NEG_POOL, sep="\t").dropna(subset=["iptm"])
    assert set(d.label.unique()) == {0}, "the decoy pool is supposed to carry no positives"
    return d[["label", "epitope", "iptm"]]


def discrimination(d: pd.DataFrame) -> dict:
    """Raw-ipTM discrimination within a labelled set: pooled, and epitope-matched."""
    out = dict(n=len(d), n_pos=int((d.label == 1).sum()), n_neg=int((d.label == 0).sum()),
               pooled_auc=float(roc_auc_score(d.label, d.iptm)))
    within, nep = [], 0
    for _, g in d.groupby("epitope"):
        if (g.label == 1).sum() >= MIN_CLASS and (g.label == 0).sum() >= MIN_CLASS:
            within.append(roc_auc_score(g.label, g.iptm))
            nep += 1
    out["within_auc"] = float(np.mean(within)) if within else float("nan")
    out["n_epitopes_matched"] = nep
    return out


def macro_at_threshold(pos: pd.DataFrame, neg: np.ndarray, t: float) -> np.ndarray:
    """Per-peptide standardised partial AUC of 1{ipTM > t}: IMMREP25 positives vs one pool."""
    nb = (neg > t).astype(float)
    vals = []
    for _, g in pos.groupby("peptide"):
        s = np.r_[(g.iptm.to_numpy() > t).astype(float), nb]
        y = np.r_[np.ones(len(g)), np.zeros(len(nb))]
        order, starts = E.tie_groups(s)
        vals.append(E.pauc01(y, order, starts))
    return np.array(vals)


def main():
    # ---- 1. which kind of non-binding does ipTM see? ----
    rows = []
    for name, kind, path in SETS:
        d = load_set(path)
        r = discrimination(d)
        r.update(dataset=name, negatives=kind)
        rows.append(r)
    disc = pd.DataFrame(rows)[["dataset", "negatives", "n", "n_pos", "n_neg", "pooled_auc",
                               "within_auc", "n_epitopes_matched"]]
    disc.to_csv(os.path.join(RESULTS, "iptm_discrimination.csv"), index=False)
    print("=== what kind of non-binding does interface confidence register? ===")
    print(disc.to_string(index=False))
    bio = float(disc[disc.negatives == "biological"].pooled_auc.iloc[0])
    comb = disc[disc.negatives == "combinatorial"].pooled_auc.to_numpy()
    print("\nipTM separates BIOLOGICAL non-binding at AUC %.4f, but COMBINATORIAL non-binding at "
          "only %s -- and IMMREP25's negatives are entirely combinatorial."
          % (bio, " / ".join("%.4f" % c for c in comb)))

    # ---- 2. what is thresholded ipTM worth on the benchmark's own metric? ----
    pos = pd.read_csv(POS)
    assert len(pos) == 1000, len(pos)
    pool = load_pool()
    neg = pool.iptm.to_numpy()
    peps = sorted(pos.peptide.unique())
    grid = np.quantile(np.r_[pos.iptm.to_numpy(), neg], np.linspace(0.01, 0.999, N_GRID))

    per_t = np.array([macro_at_threshold(pos, neg, float(t)) for t in grid])   # N_GRID x n_pep
    macro = per_t.mean(axis=1)
    j = int(np.argmax(macro))
    t_best, macro_shared = float(grid[j]), float(macro[j])
    own = per_t.max(axis=0)                       # each peptide's own best threshold: an ORACLE
    macro_own = float(own.mean())
    k = int(np.argmax(own))

    u = float((neg > t_best).mean())
    v_got = float((pos.iptm.to_numpy() > t_best).mean())
    v_need = required_tpr(u)

    pd.DataFrame(dict(peptide=peps, shared_threshold=per_t[j], own_threshold_oracle=own)).to_csv(
        os.path.join(RESULTS, "iptm_predictor.csv"), index=False)

    print("\n=== thresholded ipTM on the one-versus-external-pool task (%d positives, %d pool) ==="
          % (len(pos), len(neg)))
    print("single shared threshold %.4f: macro-AUC_%.1f %.4f  (%d of %d peptides >= %.2f)"
          % (t_best, MAX_FPR, macro_shared, int((per_t[j] >= AUC_BEST).sum()), len(peps), AUC_BEST))
    print("per-epitope thresholds (ORACLE, each peptide its own cut): %.4f  (%d of %d >= %.2f)"
          % (macro_own, int((own >= AUC_BEST).sum()), len(peps), AUC_BEST))
    print("best single peptide: %s at %.4f" % (peps[k], own[k]))
    print("at the shared cut the pool's false-positive rate is %.4f, where reaching %.2f needs "
          "mean TPR %.4f; ipTM delivers %.4f" % (u, AUC_BEST, v_need, v_got))
    print("NOTE the pool is external, so every figure here is an OPTIMISTIC bound on the "
          "benchmark's own task: its negatives are other epitopes' genuine binders.")

    macros = {
        "ippBioAuc": "%.4f" % bio,
        "ippBioSet": "TCRvdb",
        "ippBioNpos": "%d" % int(disc[disc.negatives == "biological"].n_pos.iloc[0]),
        "ippBioNneg": "%d" % int(disc[disc.negatives == "biological"].n_neg.iloc[0]),
        "ippCombLo": "%.4f" % float(comb.min()),
        "ippCombHi": "%.4f" % float(comb.max()),
        "ippCombGap": "%.4f" % (bio - float(comb.max())),
        "ippNsets": "%d" % len(SETS),
        "ippNpos": "%d" % len(pos),
        "ippNneg": "%d" % len(neg),
        "ippNpep": "%d" % len(peps),
        "ippNgrid": "%d" % N_GRID,
        "ippThresh": "%.4f" % t_best,
        "ippMacroShared": "%.4f" % macro_shared,
        "ippMacroOwn": "%.4f" % macro_own,
        "ippNaboveShared": "%d" % int((per_t[j] >= AUC_BEST).sum()),
        "ippNaboveOwn": "%d" % int((own >= AUC_BEST).sum()),
        "ippBestPep": peps[k],
        "ippBestPepAuc": "%.4f" % own[k],
        "ippFpr": "%.4f" % u,
        "ippTprNeed": "%.4f" % v_need,
        "ippTprGot": "%.4f" % v_got,
    }
    with open(os.path.join(ADAT, "iptm_predictor_macros.tex"), "w") as fh:
        for kk, vv in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (kk, vv))
    print("\nwrote results/iptm_discrimination.csv, results/iptm_predictor.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "iptm_predictor_macros.tex"), len(macros)))


def demo():
    """Self-check on synthetic data, no files read.

    The load-bearing pair is 1 and 2: the closed form must agree with the empirical partial AUC
    for a binary score, and the value must not move when the negative pool is enlarged. Together
    they are what licenses scoring IMMREP25's positives against a pool of a different size.
    """
    rng = np.random.default_rng(0)
    for u, v in [(0.05, 0.30), (0.10, 0.55), (0.20, 0.90), (0.001, 0.05)]:
        npos, nneg = 200, 1000
        npos_hit, nneg_hit = int(round(v * npos)), int(round(u * nneg))
        s = np.r_[np.ones(npos_hit), np.zeros(npos - npos_hit),
                  np.ones(nneg_hit), np.zeros(nneg - nneg_hit)]
        y = np.r_[np.ones(npos), np.zeros(nneg)]
        order, starts = E.tie_groups(s)
        got = E.pauc01(y, order, starts)
        want = std_pauc_binary(u, v)
        # 1. the closed form is the empirical value
        assert abs(got - want) < 1e-12, (
            "closed form %.12f != empirical %.12f at (u,v)=(%.3f,%.3f)" % (want, got, u, v))
        # 2. and it is invariant to the size of the negative pool at fixed rate u
        s3 = np.r_[s[:npos], np.tile(s[npos:], 3)]
        y3 = np.r_[np.ones(npos), np.zeros(3 * nneg)]
        o3, st3 = E.tie_groups(s3)
        assert abs(E.pauc01(y3, o3, st3) - got) < 1e-12, "tripling the pool moved the score"
        # 3. the inversion round-trips
        need = required_tpr(u, target=want)
        assert abs(need - v) < 1e-9, "required_tpr(%.3f) gave %.6f, not %.6f" % (u, need, v)

    # 4. sanity on the geometry: at FPR above the integration limit a binary score cannot reach
    #    the leader's score however good its TPR, because the whole window lies on the first
    #    segment and the standardised area is capped by the segment's slope
    assert std_pauc_binary(0.5, 1.0) < 1.0, "a coarse threshold should not saturate the metric"
    assert std_pauc_binary(0.0, 1.0) == 1.0, "a perfect separation should saturate it"
    # 5. a score with no discrimination sits exactly at chance
    assert abs(std_pauc_binary(0.3, 0.3) - 0.5) < 1e-12, "u == v should be chance"

    # 6. the per-peptide machinery agrees with sklearn on a continuous score
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=400) + y
    order, starts = E.tie_groups(x)
    assert abs(E.pauc01(y.astype(float), order, starts)
               - roc_auc_score(y, x, max_fpr=MAX_FPR)) < 1e-9, "disagrees with sklearn"

    print("iptm_predictor.demo OK  (binary-score closed form = validation_efficiency.auc_ceiling "
          "at f=1-v, g=u, to 1e-12 over four operating points; invariant to a 3x larger negative "
          "pool; inversion round-trips; u==v gives %.4f)" % std_pauc_binary(0.3, 0.3))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
