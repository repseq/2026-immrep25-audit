#!/usr/bin/env python
# 2026-09-13  Does the cohort ordering survive the probe's free parameters?
#
# The homology probe has three arguments a reviewer can reasonably suspect of doing the work:
# the continuity pseudocount, the substitution radius, and the minimum number of records an
# epitope needs to be scored. This sweeps all three and reports the ordering in every cell.
# No new estimator is involved -- these are per_epitope_sn's own parameters -- which is the
# point: the cheapest honest answer to "is this an artefact of your thresholds" is the grid.
#
# Two things this module is careful about, because both could flatter the result:
#
#   * pseudo = 0 does not give an unbiased ratio, it gives an UNDEFINED one. With no
#     cross-epitope neighbours the denominator is zero, so S/N is inf or NaN and
#     dataset_sn drops that epitope -- a selection effect, since the dropped epitopes are
#     exactly the least convergent ones. We therefore record n_finite beside n_ep in every
#     cell and exclude pseudo = 0 from the headline claim, reporting it separately as the
#     boundary case it is.
#
#   * a ratio must be read at MATCHED parameters. Comparing cohorts across different cells
#     would let the radius do the work, so every comparison below pairs cells with identical
#     (min_n, pseudo, d).
#
# What the grid actually shows, and the reason this module exists rather than a single number:
# the ladder splits. IMMREP25 sits below VDJdb(HQ) in every cell, but it does NOT sit below
# IMMREP22 or VDJdb(LQ) in every cell -- consistent with the Welch bounds, which do not
# separate those pairs on homology either, and with the manuscript resting the seen-versus-
# unseen separation on the embedding probe rather than on homology. The module reports the
# fragility rather than burying it.
#
# Run: python src/sensitivity.py      (--demo for the self-check)
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

from cohorts import build_cohorts                     # noqa: E402
from homology import dataset_sn, per_epitope_sn       # noqa: E402

PSEUDO = (0.0, 0.1, 0.5, 1.0)
MIN_N = (10, 20, 30, 50)
RADII = (1, 2, 3)
REFERENCE = "immrep25_pos"
COMPARATORS = ("vdjdb_hq", "tcrvdb_true", "vdjdb_lq", "immrep22_true", "pairseq_mock",
               "mlr_prolif", "airr_control", "olga_random")
SEED = 1


def grid(coh) -> pd.DataFrame:
    rows = []
    names = [REFERENCE, *COMPARATORS]
    for name in names:
        if name not in coh:
            continue
        lg = coh[name]["long"]
        for chain in ("A", "B"):
            lc = lg[lg.chain == chain]
            if lc.empty:
                continue
            for mn in MIN_N:
                for ps in PSEUDO:
                    # pseudo=0 divides by zero for a zero-count epitope; that is the defined
                    # behaviour of the boundary case, not a failure, so silence the warning
                    # and record how many epitopes survive it.
                    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                        warnings.simplefilter("ignore", RuntimeWarning)
                        pe = per_epitope_sn(lc, max_d=max(RADII), min_n=mn, pseudo=ps, seed=SEED)
                        if not len(pe):
                            continue
                        for d in RADII:
                            s = dataset_sn(pe, d)
                            v = pe["sn%d" % d].to_numpy(float)
                            rows.append(dict(
                                cohort=name, chain=chain, min_n=mn, pseudo=ps, d=d,
                                n_ep=len(pe),
                                n_finite=int(np.sum(np.isfinite(v) & (v > 0))),
                                n_pinned=int(np.sum(np.isclose(v, 1.0, atol=1e-9))),
                                sn=s["sn"], lo=s["lo"], hi=s["hi"]))
    return pd.DataFrame(rows)


def ratios(g: pd.DataFrame, chain: str, d: int, drop_pseudo_zero: bool = True) -> pd.DataFrame:
    """IMMREP25 / comparator at matched (min_n, pseudo) cells."""
    sub = g[(g.chain == chain) & (g.d == d)]
    if drop_pseudo_zero:
        sub = sub[sub.pseudo > 0]
    ref = sub[sub.cohort == REFERENCE].set_index(["min_n", "pseudo"]).sn
    out = []
    for c in COMPARATORS:
        cmp_ = sub[sub.cohort == c].set_index(["min_n", "pseudo"]).sn
        common = ref.index.intersection(cmp_.index)
        if not len(common):
            continue
        r = (ref.loc[common] / cmp_.loc[common]).to_numpy(float)
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        out.append(dict(chain=chain, d=d, comparator=c, cells=len(r),
                        ratio_min=r.min(), ratio_max=r.max(), ratio_median=float(np.median(r)),
                        n_immrep_higher=int(np.sum(r > 1.0)),
                        immrep_below_everywhere=bool(np.all(r < 1.0))))
    return pd.DataFrame(out)


def main():
    coh = build_cohorts()
    g = grid(coh)
    g.to_csv(os.path.join(RESULTS, "sensitivity_grid.csv"), index=False)
    print("grid: %d cells over pseudo=%s, min_n=%s, d=%s" % (len(g), PSEUDO, MIN_N, RADII))

    imm = g[(g.cohort == REFERENCE) & (g.pseudo > 0)]
    print("\nIMMREP25 S/N across the grid (pseudo>0):")
    for chain in ("A", "B"):
        s = imm[imm.chain == chain].sn
        s1 = imm[(imm.chain == chain) & (imm.d == 1)].sn
        print("  TCR%s: all cells %.2f-%.2f (n=%d); at d=1 only %.2f-%.2f (n=%d)"
              % (chain, s.min(), s.max(), len(s), s1.min(), s1.max(), len(s1)))

    rat = pd.concat([ratios(g, c, d) for c in ("A", "B") for d in RADII], ignore_index=True)
    rat.to_csv(os.path.join(RESULTS, "sensitivity_ratios.csv"), index=False)
    print("\n=== IMMREP25 / comparator at matched cells, d=1 (pseudo>0) ===")
    print(rat[rat.d == 1].to_string(index=False, float_format=lambda x: "%.3f" % x))

    robust = rat[(rat.d == 1) & rat.immrep_below_everywhere].comparator.tolist()
    fragile = rat[(rat.d == 1) & ~rat.immrep_below_everywhere & rat.comparator.isin(
        ("vdjdb_hq", "tcrvdb_true", "vdjdb_lq", "immrep22_true"))].comparator.tolist()
    print("\nIMMREP25 below in EVERY d=1 cell, both chains considered separately: %s"
          % sorted(set(robust)))
    print("ordering not robust against: %s" % sorted(set(fragile)))

    print("\n=== pseudocount boundary: epitopes lost to an undefined ratio at pseudo=0 ===")
    b = g[(g.chain == "B") & (g.d == 1) & (g.min_n == 30)]
    print(b.pivot_table(index="cohort", columns="pseudo",
                        values=["n_finite", "n_pinned"]).to_string())

    hq = rat[(rat.d == 1) & (rat.comparator == "vdjdb_hq")]
    macros = {
        "sgCells": "%d" % len(g[g.pseudo > 0]),
        "sgPseudo": ", ".join("%.1f" % p for p in PSEUDO if p > 0),
        "sgMinN": ", ".join(str(m) for m in MIN_N),
        "sgRadii": ", ".join(str(d) for d in RADII),
        "sgImmLoB": "%.2f" % imm[(imm.chain == "B") & (imm.d == 1)].sn.min(),
        "sgImmHiB": "%.2f" % imm[(imm.chain == "B") & (imm.d == 1)].sn.max(),
        "sgImmLoA": "%.2f" % imm[(imm.chain == "A") & (imm.d == 1)].sn.min(),
        "sgImmHiA": "%.2f" % imm[(imm.chain == "A") & (imm.d == 1)].sn.max(),
        "sgHqRatioMax": "%.2f" % hq.ratio_max.max(),
        "sgHqCells": "%d" % int(hq.cells.sum()),
        "sgHqNhigher": "%d" % int(hq.n_immrep_higher.sum()),
    }
    with open(os.path.join(ADAT, "sensitivity_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "sensitivity_grid.csv"))
    print("wrote %s" % os.path.join(RESULTS, "sensitivity_ratios.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "sensitivity_macros.tex"), len(macros)))


def demo():
    """Self-check: the pseudocount boundary behaves as documented, and ratios are matched."""
    coh = build_cohorts()
    lg = coh[REFERENCE]["long"]
    lc = lg[lg.chain == "B"]
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        p0 = per_epitope_sn(lc, max_d=1, min_n=30, pseudo=0.0, seed=SEED)
        p5 = per_epitope_sn(lc, max_d=1, min_n=30, pseudo=0.5, seed=SEED)
    v0, v5 = p0.sn1.to_numpy(float), p5.sn1.to_numpy(float)
    m1, l1 = p0.m1.to_numpy(float), p0.l1.to_numpy(float)
    assert len(v0) == len(v5), (len(v0), len(v5))
    # The two boundary sets are DIFFERENT, and conflating them is easy: S/N is
    # (m1/P_self)/(l1/P_non), so pseudo=0 makes it undefined-or-zero whenever the numerator OR
    # the denominator count vanishes, while a nonzero pseudocount pins it at exactly 1 only
    # when BOTH do. On IMMREP25 TCRbeta that is 14 epitopes against 4.
    dropped = ~np.isfinite(v0) | (v0 <= 0)
    pinned = np.isclose(v5, 1.0, atol=1e-9)
    assert np.all(dropped == ((m1 == 0) | (l1 == 0))), "dropped-at-pseudo-0 is (m1==0 or l1==0)"
    assert np.all(pinned == ((m1 == 0) & (l1 == 0))), "pinned-at-1 is (m1==0 and l1==0)"
    assert int(dropped.sum()) > int(pinned.sum()) > 0, (dropped.sum(), pinned.sum())
    n_dropped, n_pinned = int(dropped.sum()), int(pinned.sum())
    # a ratio table must only pair cells with identical parameters
    g = pd.DataFrame([
        dict(cohort=REFERENCE, chain="B", min_n=30, pseudo=0.5, d=1, sn=2.0,
             n_ep=20, n_finite=20, n_pinned=4, lo=1.0, hi=3.0),
        dict(cohort="vdjdb_hq", chain="B", min_n=30, pseudo=0.5, d=1, sn=10.0,
             n_ep=83, n_finite=83, n_pinned=1, lo=8.0, hi=12.0),
    ])
    r = ratios(g, "B", 1)
    assert len(r) == 1 and abs(r.ratio_min.iloc[0] - 0.2) < 1e-9, r
    assert bool(r.immrep_below_everywhere.iloc[0])
    print("sensitivity.demo OK  (pseudo=0 drops %d of %d epitopes as undefined -- exactly those "
          "with no within- OR no cross-epitope pair -- while pseudo>0 pins only the %d with "
          "neither at S/N=1; matched-cell ratio verified)"
          % (n_dropped, len(v0), n_pinned))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
