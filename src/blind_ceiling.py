#!/usr/bin/env python
# 2026-09-13  The hard ceiling on macro-AUC_0.1 for a peptide-blind (TCR + HLA only) scorer.
#
# IMMREP25 forms its negatives by re-pairing each receptor with the other nine peptides OF ITS
# OWN allele, so for every peptide p the 500 evaluated rows correspond ONE-TO-ONE to the 500
# receptors of that allele: 50 whose cognate peptide is p, and 450 whose cognate peptide is one
# of the other nine. A scorer that reads only the receptor and the allele therefore assigns one
# number per receptor and induces a SINGLE SHARED RANKING of those 500, reused for all ten
# per-peptide evaluations. It cannot rank the same receptor differently for different peptides.
#
# That single constraint caps the benchmark's own metric. Over FPR <= f the window admits only
# the first f*m*(K-1) non-members of a pool -- 45 of 450 here -- which is FEWER than one pool's
# m = 50 members. So placing one pool's 50 receptors above everything else saturates that pool
# at 1.0 while spending every other pool's entire false-positive budget on those same 50 rows,
# pinning the other nine at the metric's floor. Hence
#
#     ceiling = [1 + (K-1) * floor] / K,    floor = 1/2 * (1 - f/(2-f)) = 9/19 at f = 0.1,
#
# which is exactly 10/19 ~ 0.5263 for K = 10. The achievable band for ANY peptide-blind scorer
# is therefore [9/19, 10/19], of width 1/19, and the reported best submission (0.60) lies above
# it -- so that score cannot have been produced without reading the peptide.
#
# The block construction relies on m > f*m*(K-1), i.e. f < 1/(K-1). At f = 0.1 and K = 10 that
# is 0.1 < 0.111: IMMREP25 sits just inside the regime where one pool's block exactly blocks
# the rest. One more peptide per allele and the argument would need redoing, so the condition
# is asserted rather than assumed.
#
# The ceiling is an ORACLE bound: attaining it requires knowing which receptors form a pool,
# which no peptide-blind model is given. It is an upper bound on what such a model could ever
# reach, not a claim that one does.
#
# NOTE this bounds a scorer that emits one number per receptor. A model REFIT SEPARATELY FOR
# EACH PEPTIDE (as germline_baseline.evaluate does) is not bounded by it: refitting per peptide
# uses the peptide's identity to select a ranking, so it produces K different rankings and can
# exceed 10/19. That is why the in-distribution germline value sits above this ceiling, and the
# distinction is load-bearing -- peptide-sequence-blind is not the same as peptide-blind.
#
# Run: python src/blind_ceiling.py      (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

MAX_FPR = 0.1
AUC_BEST = 0.60          # best of 126 IMMREP25 submissions (Richardson 2026)
N_RESTART = 30
N_STEP = 1500
SEED = 0


def floor_value(f: float = MAX_FPR) -> float:
    """The McClish-standardised partial AUC of a ranking that places no positive inside the
    region: raw area 0 maps to 1/2 * (1 - (f/2)/(1 - f/2)) = 9/19 at f = 0.1."""
    return 0.5 * (1.0 - (f / 2) / (1.0 - f / 2))


def ceiling(k_pools: int, m: int, f: float = MAX_FPR) -> float:
    """Supremum of the macro over all single shared rankings: one pool perfect, the rest floored."""
    assert m > f * m * (k_pools - 1), (
        "one pool's block no longer exhausts the others' budget at f=%g, K=%d; the construction "
        "and hence this formula must be re-derived" % (f, k_pools))
    return (1.0 + (k_pools - 1) * floor_value(f)) / k_pools


def macro(labels: np.ndarray, peps, f: float = MAX_FPR) -> float:
    """Macro-mean standardised partial AUC of the ranking implied by `labels` read top-first.

    `labels[i]` is the pool of the receptor at rank i. Scores are strictly decreasing in rank,
    so there are no ties and the ranking is exactly the array order.
    """
    s = -np.arange(len(labels), dtype=float)
    return float(np.mean([roc_auc_score((labels == p).astype(int), s, max_fpr=f) for p in peps]))


def block(peps, m: int) -> np.ndarray:
    """One pool at a time: the construction that attains the ceiling."""
    return np.array([p for p in peps for _ in range(m)])


def interleave(peps, m: int) -> np.ndarray:
    """Round-robin: every pool equally represented at the top. The natural rival construction."""
    return np.array([p for _ in range(m) for p in peps])


def window(k_pools: int, m: int, f: float = MAX_FPR) -> int:
    """Ranks beyond which no pool can gain: its window closes at its (f*m*(K-1))-th non-member,
    which sits at rank f*m*(K-1) + m at the latest. Only this prefix affects the objective."""
    return int(f * m * (k_pools - 1)) + m


def search(peps, m: int, rng: np.random.Generator, n_restart: int = N_RESTART,
           n_step: int = N_STEP, f: float = MAX_FPR):
    """Hill-climb the macro over shared rankings, restricted to the prefix that can matter.

    Swaps are proposed inside the window, or between the window and the tail; anything deeper is
    provably irrelevant (see `window`). Starts from the block, the interleave and random
    arrangements so the search is not anchored on the construction it is meant to test.
    """
    W = window(len(peps), m, f)
    best, best_lab = -np.inf, None
    starts = [block(peps, m), interleave(peps, m)]
    while len(starts) < n_restart:
        lab = block(peps, m).copy()
        rng.shuffle(lab)
        starts.append(lab)
    for lab0 in starts:
        lab = lab0.copy()
        cur = macro(lab, peps, f)
        for _ in range(n_step):
            i = int(rng.integers(0, W))
            j = int(rng.integers(0, len(lab)))
            if lab[i] == lab[j]:
                continue
            lab[i], lab[j] = lab[j], lab[i]
            val = macro(lab, peps, f)
            if val > cur:
                cur = val
            else:
                lab[i], lab[j] = lab[j], lab[i]      # reject
        if cur > best:
            best, best_lab = cur, lab.copy()
    return best, best_lab


def empirical_best() -> tuple[float, str]:
    """Best macro actually reached by the committed epitope-blind receptor scores, as the
    realizable anchor beneath the oracle ceiling. Reuses epitope_free's own score battery."""
    import pandas as pd
    import epitope_free as E
    bench = pd.read_csv(BENCH, sep="\t")
    rec = bench[bench.label == 1][["tcra_cdr3", "tcrb_cdr3", "peptide", "hla"]].reset_index(drop=True)
    S, _ = E.build_scores(rec)
    best, who = -np.inf, ""
    for name, v in S.items():
        vals = []
        for _, g in rec.groupby("hla"):
            idx = g.index.to_numpy()
            vals.extend(E.per_peptide_pauc(v[idx], g.peptide.to_numpy(),
                                           sorted(g.peptide.unique())).tolist())
        mac = float(np.mean(vals))
        if mac > best:
            best, who = mac, name
    return best, who


def geometry() -> pl.DataFrame:
    """The benchmark's own per-allele geometry, read rather than assumed."""
    return (pl.read_csv(BENCH, separator="\t")
            .filter(pl.col("label") == 1)
            .group_by("hla")
            .agg(n_pools=pl.col("peptide").n_unique(), n_receptors=pl.len())
            .sort("hla"))


def main():
    rng = np.random.default_rng(SEED)
    geo = geometry()
    print("=== IMMREP25 per-allele geometry (read from the release) ===")
    print(geo)

    rows = []
    for r in geo.to_dicts():
        K, N = r["n_pools"], r["n_receptors"]
        m = N // K
        assert K * m == N, "pools are not equal-sized; the closed form assumes they are"
        peps = np.arange(K)
        fl, ce = floor_value(), ceiling(K, m)
        bl = macro(block(peps, m), peps)
        il = macro(interleave(peps, m), peps)
        se, _ = search(peps, m, rng)
        rows.append(dict(hla=r["hla"], n_pools=K, pool_size=m, window=window(K, m),
                         floor=fl, analytic_ceiling=ce, block=bl, interleave=il, search_best=se))
    t = pl.DataFrame(rows)
    with pl.Config(tbl_width_chars=200, float_precision=6):
        print("\n=== ceiling on macro-AUC_0.1 for a single shared (peptide-blind) ranking ===")
        print(t)
    t.write_csv(os.path.join(RESULTS, "blind_ceiling.csv"))

    fl = float(t["floor"][0])
    ce = float(t["analytic_ceiling"][0])
    bl = float(t["block"].max())
    il = float(t["interleave"].max())      # from the table, not the loop variable it shadows
    se = float(t["search_best"].max())
    emp, who = empirical_best()
    print("\nfloor %.6f (= 9/19)   ceiling %.6f (= 10/19)   band width %.6f (= 1/19)"
          % (fl, ce, ce - fl))
    print("block construction attains %.6f; round-robin interleave gives %.6f" % (bl, il))
    print("best of %d hill-climb restarts x %d steps: %.6f  (exceeds ceiling: %s)"
          % (N_RESTART, N_STEP, se, se > ce + 1e-9))
    print("best macro reached by a committed epitope-blind score: %.4f (%s)" % (emp, who))
    print("best of 126 challenge submissions: %.2f -- above the ceiling by %.4f"
          % (AUC_BEST, AUC_BEST - ce))

    macros = {
        "blindFloor": "%.4f" % fl,
        "blindCeil": "%.4f" % ce,
        "blindCeilFrac": r"10/19",
        "blindFloorFrac": r"9/19",
        "blindBand": "%.4f" % (ce - fl),
        "blindBlock": "%.4f" % bl,
        "blindInterleave": "%.4f" % il,
        "blindSearch": "%.4f" % se,
        "blindNrestart": "%d" % N_RESTART,
        "blindNstep": "%d" % N_STEP,
        "blindWindow": "%d" % int(t["window"][0]),
        "blindEmp": "%.4f" % emp,
        "blindEmpScore": who.replace("_", ""),
        "blindLeaderGap": "%.4f" % (AUC_BEST - ce),
    }
    with open(os.path.join(ADAT, "blind_ceiling_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/blind_ceiling.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "blind_ceiling_macros.tex"), len(macros)))


def demo():
    """Self-check. The load-bearing assertion is the last one: a hill-climbing search must never
    exceed the analytic ceiling. If the derivation is wrong, that is where it fails -- the same
    class of error as the partial-AUC label-noise ceiling, which was also asserted by analogy
    before its own self-check caught it."""
    K, m = 10, 50
    peps = np.arange(K)
    assert abs(floor_value() - 9 / 19) < 1e-12, "floor is not 9/19 at f=0.1"
    assert abs(ceiling(K, m) - 10 / 19) < 1e-12, "ceiling is not 10/19 at K=10, f=0.1"
    # the block construction attains the ceiling exactly
    bl = macro(block(peps, m), peps)
    assert abs(bl - 10 / 19) < 1e-9, "block construction does not attain 10/19 (got %.9f)" % bl
    # one pool perfect, nine floored -- check the components, not just the mean
    s = -np.arange(K * m, dtype=float)
    lab = block(peps, m)
    per = [roc_auc_score((lab == p).astype(int), s, max_fpr=MAX_FPR) for p in peps]
    assert abs(per[0] - 1.0) < 1e-12, "the top pool should be perfectly separated"
    assert all(abs(v - 9 / 19) < 1e-12 for v in per[1:]), "the other pools should sit at the floor"
    # the round-robin rival sits at chance, well below the block
    il = macro(interleave(peps, m), peps)
    assert il < bl, "interleaving should not beat concentration"
    # only the prefix matters: permuting below the window cannot change the objective
    W = window(K, m)
    lab2 = lab.copy()
    tail = lab2[W:].copy()
    np.random.default_rng(1).shuffle(tail)
    lab2[W:] = tail
    assert abs(macro(lab2, peps) - bl) < 1e-12, "the objective depends on ranks beyond the window"
    # cross-implementation check against the repo's own tie-aware partial AUC
    import epitope_free as E
    order, starts = E.tie_groups(s)
    mine = roc_auc_score((lab == 0).astype(int), s, max_fpr=MAX_FPR)
    theirs = E.pauc01((lab == 0).astype(float), order, starts)
    assert abs(mine - theirs) < 1e-9, "sklearn and epitope_free.pauc01 disagree (%.9f vs %.9f)" % (
        mine, theirs)
    # THE check: search must not beat the bound
    se, _ = search(peps, m, np.random.default_rng(SEED), n_restart=6, n_step=400)
    assert se <= 10 / 19 + 1e-9, "search exceeded the analytic ceiling (%.9f > %.9f)" % (se, 10 / 19)
    print("blind_ceiling.demo OK  (floor 9/19 = %.6f, ceiling 10/19 = %.6f attained exactly by "
          "the block; interleave %.6f; 6x400-step search reached %.6f without exceeding it; "
          "sklearn agrees with epitope_free.pauc01)" % (floor_value(), 10 / 19, il, se))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
