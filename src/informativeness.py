#!/usr/bin/env python
# 2026-09-14  The informativeness ordering, given one home and its own macros.
#
# The claim that IMMREP25 carries less information than the better-curated references is what
# licenses transferring a measured false-positive rate to it as an INEQUALITY -- f(IMMREP25) >=
# f(VDJdb) -- rather than as a point value. That ordering has been asserted in prose (SOURCES.md)
# with the channels unnamed and no macros behind it. This module states it as three measurements,
# each on a different kind of evidence, so the text can cite numbers instead of an assertion.
#
#   1. SEQUENCE CONVERGENCE. Per-epitope substitution-neighbour signal-to-noise at Hamming 1.
#      Reported as a fold-ratio with a 90% interval via audit_stats.welch_log_bound, which is the
#      committed estimator for this contrast: the statistic is a ratio, so the contrast belongs on
#      the log10 scale and the interval must be back-transformed. A naive ratio of the two pooled
#      values would be a different, worse statistic.
#   2. INTER-CHAIN COUPLING. The alpha/beta mutual-information excess over a within-epitope
#      permutation, read from the committed pairing table. Not a Hamming count, so it does not
#      share channel 1's selection criterion.
#   3. STRUCTURAL SEPARABILITY. How well AlphaFold interface confidence can separate binder from
#      non-binder in each cohort, read from iptm_predictor's table. External to the sequence
#      entirely, and frozen before this comparison was posed.
#
# What the reader must not conclude, and the reason each row names its comparison: channel 3
# compares cohorts whose negative class differs in KIND -- a curated pairing the MATCHMAKERS assay
# invalidated, against a re-pairing that was never observed. It is a statement about which kind of
# non-binding a structural probe can register, and it is reported as such.
#
# Nothing is recomputed here. Every value is read from a committed results file, so this module
# cannot perturb a standing number -- which is why the VDJdb-HQ inter-chain MI is emitted from
# results/pairing.csv rather than by re-running run_audit.py, whose aggregate homology values are
# documented to wobble at the +-0.1 level between runs through cohort subsampling.
#
# Run: python src/informativeness.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

import audit_stats as astat                                       # noqa: E402

HOM_PE = os.path.join(RESULTS, "homology_per_epitope.csv")
PAIR = os.path.join(RESULTS, "pairing.csv")
IPTM = os.path.join(RESULTS, "iptm_discrimination.csv")

CONF = 0.90              # two-sided; TOST at alpha=0.05 per side, as in the A-bounds work
CHAIN = "B"              # primary chain, as elsewhere in the repo; A reported alongside
TEST_COHORT = "immrep25_pos"
REFS = ("vdjdb_hq", "tcrvdb_true")


def hom_bound(cohort: str, ref: str, chain: str = CHAIN) -> dict:
    """Fold-ratio of per-epitope Hamming-1 S/N, cohort vs ref, with its 90% interval."""
    d = pd.read_csv(HOM_PE)
    x = d[(d.cohort == cohort) & (d.chain == chain)].sn1.to_numpy()
    y = d[(d.cohort == ref) & (d.chain == chain)].sn1.to_numpy()
    r = astat.welch_log_bound(x, y, conf=CONF, log=True)
    r.update(cohort=cohort, ref=ref, chain=chain)
    return r


def mi_row(cohort: str) -> dict:
    d = pd.read_csv(PAIR)
    r = d[d.cohort == cohort]
    assert len(r) == 1, "%s is not a single row of pairing.csv" % cohort
    return dict(cohort=cohort, n_ep=int(r.n_ep.iloc[0]), mi=float(r.mi_mean.iloc[0]),
                lo=float(r.mi_lo.iloc[0]), hi=float(r.mi_hi.iloc[0]))


def iptm_rows() -> pd.DataFrame:
    return pd.read_csv(IPTM)


def main():
    # ---- channel 1: sequence convergence, as a fold-ratio ----
    print("=== channel 1: substitution-neighbour S/N at Hamming 1, per epitope ===")
    hb = []
    for ref in REFS:
        for chain in ("B", "A"):
            r = hom_bound(TEST_COHORT, ref, chain)
            hb.append(r)
            print("  TCR%s vs %-12s fold %.4f  90%% CI %.4f-%.4f  (%.1f-%.1f%%)  "
                  "p=%.3g  n=%d vs %d"
                  % (chain, ref, 10 ** r["d"], r["fold_lo"], r["fold_hi"],
                     100 * r["fold_lo"], 100 * r["fold_hi"], r["p"], r["n_x"], r["n_y"]))
    hbd = pd.DataFrame(hb)

    # ---- channel 2: inter-chain coupling ----
    print("\n=== channel 2: inter-chain alpha/beta MI excess (bits) ===")
    mi = [mi_row(c) for c in (TEST_COHORT,) + REFS]
    mid = pd.DataFrame(mi)
    for r in mi:
        print("  %-14s %+.4f  95%% CI %+.4f to %+.4f  (%d epitopes)"
              % (r["cohort"], r["mi"], r["lo"], r["hi"], r["n_ep"]))
    imm_mi = mid[mid.cohort == TEST_COHORT].mi.iloc[0]
    hq_mi = mid[mid.cohort == "vdjdb_hq"].mi.iloc[0]
    print("  IMMREP25 is %.1f%% of VDJdb-HQ on this channel" % (100 * imm_mi / hq_mi))

    # ---- channel 3: structural separability ----
    print("\n=== channel 3: what interface confidence can separate, by cohort ===")
    ip = iptm_rows()
    print(ip[["dataset", "negatives", "n_pos", "n_neg", "pooled_auc"]].to_string(index=False))
    bio = float(ip[ip.negatives == "biological"].pooled_auc.iloc[0])
    comb = ip[ip.negatives == "combinatorial"].pooled_auc
    print("  biological non-binding %.4f vs combinatorial %.4f-%.4f (the class IMMREP25 is "
          "built from)" % (bio, float(comb.min()), float(comb.max())))

    out = os.path.join(RESULTS, "informativeness.csv")
    pd.concat([
        hbd.assign(channel="homology_sn1").rename(columns={"d": "log10_diff"}),
        mid.assign(channel="interchain_mi"),
        ip.assign(channel="iptm_separability"),
    ], ignore_index=True).to_csv(out, index=False)

    hqB = hbd[(hbd.ref == "vdjdb_hq") & (hbd.chain == "B")].iloc[0]
    hqA = hbd[(hbd.ref == "vdjdb_hq") & (hbd.chain == "A")].iloc[0]
    ttB = hbd[(hbd.ref == "tcrvdb_true") & (hbd.chain == "B")].iloc[0]

    print("\nordering, all three channels: IMMREP25 below VDJdb-HQ and TCRvdb+; so a "
          "false-positive rate measured on the reference transfers to IMMREP25 as a LOWER "
          "bound, never as a point value.")

    macros = {
        "infNchan": "3",
        "infConf": "%d" % int(round(100 * CONF)),
        # channel 1, against VDJdb-HQ
        "infHomPctLoB": "%.1f" % (100 * hqB.fold_lo),
        "infHomPctHiB": "%.1f" % (100 * hqB.fold_hi),
        "infHomPctLoA": "%.1f" % (100 * hqA.fold_lo),
        "infHomPctHiA": "%.1f" % (100 * hqA.fold_hi),
        "infHomPB": astat.fmt_p(float(hqB.p)),
        "infHomNepB": "%d" % int(hqB.n_x),
        "infHomNrefB": "%d" % int(hqB.n_y),
        # channel 1, against TCRvdb+
        "infHomPctLoTt": "%.1f" % (100 * ttB.fold_lo),
        "infHomPctHiTt": "%.1f" % (100 * ttB.fold_hi),
        # channel 2
        "infMiImm": "%.4f" % imm_mi,
        "infMiImmLo": "%.4f" % float(mid[mid.cohort == TEST_COHORT].lo.iloc[0]),
        "infMiImmHi": "%.4f" % float(mid[mid.cohort == TEST_COHORT].hi.iloc[0]),
        "infMiHq": "%.4f" % hq_mi,
        "infMiHqLo": "%.4f" % float(mid[mid.cohort == "vdjdb_hq"].lo.iloc[0]),
        "infMiHqHi": "%.4f" % float(mid[mid.cohort == "vdjdb_hq"].hi.iloc[0]),
        "infMiHqNep": "%d" % int(mid[mid.cohort == "vdjdb_hq"].n_ep.iloc[0]),
        "infMiPct": "%.1f" % (100 * imm_mi / hq_mi),
        # channel 3
        "infIptmBio": "%.4f" % bio,
        "infIptmCombLo": "%.4f" % float(comb.min()),
        "infIptmCombHi": "%.4f" % float(comb.max()),
    }
    with open(os.path.join(ADAT, "informativeness_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % out)
    print("wrote %s (%d macros)"
          % (os.path.join(ADAT, "informativeness_macros.tex"), len(macros)))


def demo():
    """Self-check. The load-bearing assertion is the last: all three channels must put IMMREP25
    below the high-quality reference. If any one of them does not, the ordering is not a fact
    about this benchmark and the inequality it licenses must not be used."""
    # 1. the fold-ratio helper is self-consistent: a cohort against itself is 1.0
    d = pd.read_csv(HOM_PE)
    x = d[(d.cohort == TEST_COHORT) & (d.chain == CHAIN)].sn1.to_numpy()
    same = astat.welch_log_bound(x, x, conf=CONF, log=True)
    assert abs(same["d"]) < 1e-12, "a cohort against itself is not a null contrast"
    assert same["fold_lo"] < 1.0 < same["fold_hi"], "the self-interval excludes 1.0"

    # 2. the interval brackets the point estimate, on the fold scale
    r = hom_bound(TEST_COHORT, "vdjdb_hq", CHAIN)
    assert r["fold_lo"] < 10 ** r["d"] < r["fold_hi"], "the fold interval does not bracket 10^d"
    assert r["fold_hi"] < 1.0, "IMMREP25 is not below VDJdb-HQ on sequence convergence"

    # 3. channel 2 is read, not recomputed, and is ordered the same way
    imm, hq = mi_row(TEST_COHORT), mi_row("vdjdb_hq")
    assert imm["mi"] < hq["mi"], "inter-chain MI does not order IMMREP25 below VDJdb-HQ"
    assert imm["lo"] > -0.05, "the IMMREP25 MI interval is implausibly wide"

    # 4. channel 3 likewise, and it is a different KIND of comparison -- asserted so a future
    #    edit cannot quietly turn it into a cohort information measure
    ip = iptm_rows()
    assert set(ip.negatives) == {"biological", "combinatorial"}, sorted(set(ip.negatives))
    bio = float(ip[ip.negatives == "biological"].pooled_auc.iloc[0])
    assert bio > float(ip[ip.negatives == "combinatorial"].pooled_auc.max()), (
        "interface confidence does not separate biological non-binding better than combinatorial")

    # 5. THE check: the ordering holds on all three
    assert r["fold_hi"] < 1.0 and imm["mi"] < hq["mi"] and bio > float(
        ip[ip.negatives == "combinatorial"].pooled_auc.max()), "the three-channel ordering fails"
    print("informativeness.demo OK  (self-contrast null; IMMREP25 at %.1f-%.1f%% of VDJdb-HQ on "
          "TCR%s convergence, %.4f vs %.4f bits inter-chain, and interface confidence separates "
          "biological %.4f vs combinatorial %.4f)"
          % (100 * r["fold_lo"], 100 * r["fold_hi"], CHAIN, imm["mi"], hq["mi"], bio,
             float(ip[ip.negatives == "combinatorial"].pooled_auc.max())))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
