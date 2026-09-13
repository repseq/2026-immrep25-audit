#!/usr/bin/env python
# 2026-09-13  Is the publicity result circular?
#
# The audit defines an IMMREP25 positive as PUBLIC when either chain lies within Hamming 1 of
# a reference receptor, then shows the homology signal-to-noise ratio collapses once the
# public fraction is removed. A reviewer objects, correctly in principle, that the selection
# criterion and the outcome statistic are both Hamming-neighbour quantities, so the collapse
# may partly be a dependence between them rather than a fact about publicity. This module
# answers that with the controls the objection calls for.
#
#   1. RANDOM REMOVAL. Drop the same NUMBER of receptors at random and rescore. This is the
#      null for "removal per se". Evaluated on the raw within-epitope neighbour-pair COUNT
#      rather than on S/N: with zero pairs the exposure-proportional pseudocount pins S/N at
#      exactly 1 with a zero-width interval, so a percentile of that is a percentile of a
#      point mass. The count has an exact null and a Poisson bound.
#
#   2. MATCHED REMOVAL. Drop NON-public receptors matched to the public ones on the
#      covariates publicity correlates with. The matching deliberately EXCLUDES the observed
#      neighbourhood degree: a receptor's within-pool Hamming-1 degree is the numerator of the
#      outcome statistic, so matching on it removes the outcome by construction and
#      guarantees a null -- a tautology, not a control. We match instead on CDR3 length, V and
#      J gene, log Pgen and OLGA's 1-mismatch neighbourhood Pgen, the last being a property of
#      the recombination model rather than of the observed pool. That is what "matchability"
#      means here, and it is the non-circular version of the reviewer's request.
#
#   3. DEFINITION SWEEP. The publicity definition is one choice among several, so vary it:
#      exact match (d=0), d<=1 and d<=2; either-chain, per-chain and both-chain; and the
#      reference database (VDJdb alone versus VDJdb plus TCRvdb). Report the public fraction
#      and the surviving subset's pair count, rate and exact Poisson bound for each.
#
# Distances are computed by blocked numpy over equal-length buckets, NOT by extending the
# 1-wildcard index of publicity.py: a d<=2 wildcard index over ~90k references needs
# L(L-1)/2 ~ 105 patterns per sequence, i.e. millions of tuple keys and gigabytes of Python
# objects, for a comparison that is a few hundred million byte operations done properly.
#
# The orthogonal-metric control -- the same split scored by inter-chain mutual information,
# which is not a Hamming statistic at all -- is already produced by run_audit.py into
# results/publicity_pairing.csv and is not duplicated here.
#
# Run: python src/publicity_controls.py      (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
CACHE = os.path.join(REPO, "cache")

from audit_stats import geo_ci, poisson_rate_upper          # noqa: E402
from cohorts import build_cohorts                            # noqa: E402
from homology import dataset_sn, per_epitope_sn              # noqa: E402
from load_data import _pair_to_long, load_vdjdb              # noqa: E402

MIN_N = 10          # the publicity contrast's per-epitope floor, applied to BOTH arms
MAX_D = 3
N_RANDOM = 1000     # random-removal draws
N_MATCHED = 200     # matched-removal draws (its only randomness is tie-breaking)
SEED = 0
_AA = "ACDEFGHIKLMNPQRSTVWY"
_IDX = {a: i for i, a in enumerate(_AA)}


def _encode(seqs) -> dict:
    """Length-bucketed int8 matrices, so Hamming is a vectorised byte comparison."""
    out = {}
    for s in seqs:
        if isinstance(s, str) and set(s) <= set(_AA):
            out.setdefault(len(s), []).append(s)
    return {L: (v, np.array([[_IDX[c] for c in s] for s in v], dtype=np.int8))
            for L, v in out.items()}


def min_distance(queries, refs, cap: int = 2, block: int = 256) -> np.ndarray:
    """Minimum Hamming distance from each query to any equal-length reference, capped at `cap`.

    Returns cap+1 where nothing equal-length is within `cap` (or no reference of that length
    exists), so callers test `d <= threshold` without special-casing misses.
    """
    ref = _encode(refs)
    out = np.full(len(queries), cap + 1, dtype=np.int8)
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(queries):
        if isinstance(s, str) and set(s) <= set(_AA):
            by_len.setdefault(len(s), []).append(i)
    for L, idx in by_len.items():
        if L not in ref:
            continue
        R = ref[L][1]
        Q = np.array([[_IDX[c] for c in queries[i]] for i in idx], dtype=np.int8)
        best = np.full(len(idx), cap + 1, dtype=np.int16)
        for s in range(0, len(R), block):
            d = (Q[:, None, :] != R[None, s:s + block, :]).sum(axis=2)
            best = np.minimum(best, d.min(axis=1))
            if (best == 0).all():
                break
        out[idx] = np.minimum(best, cap + 1).astype(np.int8)
    return out


def _reference(coh, with_tcrvdb: bool) -> dict:
    vlong, _ = load_vdjdb()
    ref = {"A": list(vlong[vlong.chain == "A"].cdr3), "B": list(vlong[vlong.chain == "B"].cdr3)}
    if with_tcrvdb:
        for name in ("tcrvdb_true", "tcrvdb_false"):
            tp = coh[name]["paired"]
            ref["A"] += list(tp.cdr3a); ref["B"] += list(tp.cdr3b)
    return ref


def score_subset(paired: pd.DataFrame) -> dict:
    """Per-chain pair counts, exposures, rate bound and S/N for one receptor subset."""
    lg = _pair_to_long(paired)
    out = {}
    for chain in ("A", "B"):
        pe = per_epitope_sn(lg[lg.chain == chain], max_d=MAX_D, min_n=MIN_N, seed=1)
        s = dataset_sn(pe, 1)
        m1 = float(pe.m1.sum()) if len(pe) else 0.0
        ps = float(pe.P_self.sum()) if len(pe) else 0.0
        out[chain] = dict(n=len(paired), n_ep=len(pe), m1=m1, P_self=ps,
                          rate=(m1 / ps) if ps else np.nan,
                          rate_upper97=poisson_rate_upper(int(m1), ps) if ps else np.nan,
                          sn=s["sn"], lo=s["lo"], hi=s["hi"])
    return out


def _covariates(paired: pd.DataFrame) -> pd.DataFrame:
    """Matching covariates: length, V/J, log Pgen and OLGA generation degree -- never the
    observed neighbourhood degree, which is the outcome statistic's own numerator."""
    pa = pd.read_csv(os.path.join(CACHE, "pgen1mm_immrep25_pos_A.tsv"), sep="\t")
    pb = pd.read_csv(os.path.join(CACHE, "pgen1mm_immrep25_pos_B.tsv"), sep="\t")
    d = (paired.merge(pa.rename(columns={"cdr3": "cdr3a", "pgen": "pg_a", "pgen1mm": "dg_a"}),
                      on="cdr3a", how="left")
               .merge(pb.rename(columns={"cdr3": "cdr3b", "pgen": "pg_b", "pgen1mm": "dg_b"}),
                      on="cdr3b", how="left"))
    for c in ("pg_a", "pg_b", "dg_a", "dg_b"):
        v = np.log10(d[c].where(d[c] > 0))
        d["l_" + c] = v.fillna(v.median())          # keep every row: median placeholder
    d["la"] = d.cdr3a.str.len(); d["lb"] = d.cdr3b.str.len()
    d["stratum"] = (d.la // 2).astype(str) + "|" + (d.lb // 2).astype(str) + "|" + d.vb.astype(str)
    return d


def matched_public_subset(ann: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    """A PUBLIC subset, size- and covariate-matched to the novel subset.

    The contrast has to be size-matched to mean anything, and the obvious framing does not
    work: there are 684 public receptors against only 316 non-public ones, so "drop the same
    number of non-public receptors" is impossible -- the pool runs out, the surviving subset is
    larger than the novel one, and it trivially retains more neighbour pairs. (That was the
    first version of this function and it produced exactly that artefact.)

    So invert it. For each of the 316 novel receptors, draw its covariate-nearest partner from
    the 684 public ones, without replacement. The result is 316 PUBLIC receptors carrying the
    novel subset's distribution of CDR3 length, V/J gene, generation probability and OLGA
    generation degree. Both arms are then 316 receptors matched on everything publicity
    correlates with, and differ only in publicity status itself -- which is the non-circular
    test. Crucially the match excludes the observed neighbourhood degree, which is the outcome
    statistic's own numerator.

    Returns the matched public subset and the achieved covariate balance.
    """
    d = _covariates(ann)
    cols = ["l_pg_a", "l_pg_b", "l_dg_a", "l_dg_b"]
    Z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0).replace(0, 1)
    novel = d.index[~d.public_any].to_numpy()
    pool = set(d.index[d.public_any].to_numpy())
    picked = []
    for i in rng.permutation(novel):               # random order, so ties break differently
        cand = [j for j in pool if d.stratum[j] == d.stratum[i]]
        if not cand:                               # no stratum partner: fall back to the pool
            cand = list(pool)
        if not cand:
            break
        dist = np.linalg.norm(Z.loc[cand].to_numpy() - Z.loc[i].to_numpy(), axis=1)
        # break exact ties at random rather than by index order, so the draw is genuinely
        # stochastic across repeats
        best = np.flatnonzero(dist <= dist.min() + 1e-12)
        pick = cand[int(rng.choice(best))]
        picked.append(pick); pool.discard(pick)
    bal = {c: float(Z.loc[picked, c].mean() - Z.loc[novel, c].mean()) for c in cols}
    bal["n_matched"] = len(picked)
    return ann.loc[sorted(picked)], bal


def main():
    rng = np.random.default_rng(SEED)
    coh = build_cohorts()
    imm = coh["immrep25_pos"]["paired"].reset_index(drop=True)
    from publicity import annotate_publicity
    vlong, _ = load_vdjdb()
    ann = annotate_publicity(imm, vlong, [coh["tcrvdb_true"]["paired"],
                                          coh["tcrvdb_false"]["paired"]]).reset_index(drop=True)
    n_pub = int(ann.public_any.sum())
    novel = ann[~ann.public_any]
    print("IMMREP25 positives %d | public %d (%.1f%%) | novel %d"
          % (len(ann), n_pub, 100 * n_pub / len(ann), len(novel)))

    base = score_subset(ann)
    nov = score_subset(novel)
    rows = [dict(arm="all", chain=c, **base[c]) for c in ("A", "B")]
    rows += [dict(arm="public_removed", chain=c, **nov[c]) for c in ("A", "B")]

    # 1. random removal: the null for removal per se, on the COUNT
    print("\n=== random removal (%d draws of %d receptors) ===" % (N_RANDOM, n_pub))
    rnd = {c: [] for c in ("A", "B")}
    for _ in range(N_RANDOM):
        keep = ann.loc[rng.choice(len(ann), len(ann) - n_pub, replace=False)]
        s = score_subset(keep)
        for c in ("A", "B"):
            rnd[c].append(s[c]["m1"])
    for c in ("A", "B"):
        v = np.array(rnd[c]); obs = nov[c]["m1"]
        p = (np.sum(v <= obs) + 1) / (N_RANDOM + 1)
        print("  TCR%s: public removal leaves %d within-epitope pairs; random removal leaves "
              "median %.0f [%.0f-%.0f]; P(random <= observed) = %.4f"
              % (c, obs, np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5), p))
        rows.append(dict(arm="random_removed", chain=c, n=len(ann) - n_pub, n_ep=np.nan,
                         m1=float(np.median(v)), P_self=np.nan, rate=np.nan,
                         rate_upper97=np.nan, sn=np.nan, lo=float(np.percentile(v, 2.5)),
                         hi=float(np.percentile(v, 97.5)), p_vs_public=p))

    # 2. size- and covariate-matched PUBLIC subset: same size as the novel arm, matched on
    #    everything publicity correlates with, differing only in publicity status
    print("\n=== covariate-matched public subset, %d vs %d receptors (%d draws) ==="
          % (len(novel), len(novel), N_MATCHED))
    mat = {c: [] for c in ("A", "B")}
    bals = []
    for _ in range(N_MATCHED):
        sub, bal = matched_public_subset(ann, rng)
        bals.append(bal)
        s = score_subset(sub)
        for c in ("A", "B"):
            mat[c].append(s[c]["m1"])
    bal_df = pd.DataFrame(bals)
    print("  achieved balance (matched-public minus novel, in SDs of each covariate):")
    print("   " + "  ".join("%s %+.3f" % (c, bal_df[c].mean())
                            for c in ("l_pg_a", "l_pg_b", "l_dg_a", "l_dg_b")))
    print("  matched receptors per draw: %.0f" % bal_df.n_matched.mean())
    for c in ("A", "B"):
        v = np.array(mat[c]); obs = nov[c]["m1"]
        p = (np.sum(v <= obs) + 1) / (N_MATCHED + 1)
        print("  TCR%s: the matched PUBLIC subset retains median %.0f within-epitope pairs "
              "[%.0f-%.0f]; the novel subset of equal size retains %d; P(matched <= novel) = %.4f"
              % (c, np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5), obs, p))
        rows.append(dict(arm="matched_public", chain=c, n=len(novel), n_ep=np.nan,
                         m1=float(np.median(v)), P_self=np.nan, rate=np.nan,
                         rate_upper97=np.nan, sn=np.nan,
                         lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                         p_vs_public=p))

    # 3. definition sweep
    print("\n=== publicity definition sweep ===")
    sweep = []
    for with_tv in (False, True):
        ref = _reference(coh, with_tv)
        da = min_distance(list(ann.cdr3a), ref["A"], cap=2)
        db = min_distance(list(ann.cdr3b), ref["B"], cap=2)
        for d_thr in (0, 1, 2):
            for rule in ("either", "alpha", "beta", "both"):
                pa_, pb_ = da <= d_thr, db <= d_thr
                flag = {"either": pa_ | pb_, "alpha": pa_, "beta": pb_, "both": pa_ & pb_}[rule]
                sub = ann[~flag]
                if len(sub) < 20:
                    continue
                s = score_subset(sub)
                sweep.append(dict(reference="vdjdb+tcrvdb" if with_tv else "vdjdb",
                                  d=d_thr, rule=rule, frac_public=float(flag.mean()),
                                  n_novel=len(sub),
                                  m1_B=s["B"]["m1"], rate_upper97_B=s["B"]["rate_upper97"],
                                  sn_B=s["B"]["sn"], m1_A=s["A"]["m1"], sn_A=s["A"]["sn"]))
    sw = pd.DataFrame(sweep)
    print(sw.to_string(index=False, float_format=lambda x: "%.4g" % x))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "publicity_controls.csv"), index=False)
    sw.to_csv(os.path.join(RESULTS, "publicity_sweep.csv"), index=False)

    med_rnd_B = float(np.median(rnd["B"]))
    med_mat_B = float(np.median(mat["B"]))
    macros = {
        "pcNpub": "%d" % n_pub,
        "pcNnovel": "%d" % len(novel),
        "pcObsPairsB": "%d" % int(nov["B"]["m1"]),
        "pcObsPairsA": "%d" % int(nov["A"]["m1"]),
        "pcAllPairsB": "%d" % int(base["B"]["m1"]),
        "pcRandPairsB": "%.0f" % med_rnd_B,
        "pcRandLoB": "%.0f" % np.percentile(rnd["B"], 2.5),
        "pcRandHiB": "%.0f" % np.percentile(rnd["B"], 97.5),
        "pcMatchPairsB": "%.0f" % med_mat_B,
        "pcMatchLoB": "%.0f" % np.percentile(mat["B"], 2.5),
        "pcMatchHiB": "%.0f" % np.percentile(mat["B"], 97.5),
        "pcRateUpperB": "%.1e" % nov["B"]["rate_upper97"],
        "pcExposureB": "%.0f" % nov["B"]["P_self"],
        "pcNdraws": "%d" % N_RANDOM,
        "pcNmatched": "%d" % N_MATCHED,
        "pcMinN": "%d" % MIN_N,
        "pcSweepN": "%d" % len(sw),
        "pcSweepFracLo": "%.1f" % (100 * sw.frac_public.min()),
        "pcSweepFracHi": "%.1f" % (100 * sw.frac_public.max()),
        "pcSweepSnMaxB": "%.2f" % sw.sn_B.max(),
    }
    with open(os.path.join(ADAT, "publicity_controls_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "publicity_controls.csv"))
    print("wrote %s" % os.path.join(RESULTS, "publicity_sweep.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "publicity_controls_macros.tex"),
                                    len(macros)))


def demo():
    """Self-check on the distance engine, which the whole sweep rests on."""
    refs = ["CASSLAPGATNEKLFF", "CASSQDRGTGELFF"]
    q = ["CASSLAPGATNEKLFF",                  # exact
         "CASSLAPGATNEKLFY",                  # one substitution
         "CASSLAPGATNEKLYY",                  # two
         "CASSLAPGATNEKLYYY",                 # different length: no equal-length reference
         "CASSQDRGTGELFF"]
    d = min_distance(q, refs, cap=2)
    assert list(d) == [0, 1, 2, 3, 0], list(d)
    # capping: a distant sequence of matching length reports cap+1, not its true distance
    assert min_distance(["WWWWWWWWWWWWWWWW"], refs, cap=2)[0] == 3
    # a non-amino-acid query is skipped rather than crashing
    assert min_distance(["CASSLAPGATNEKLF*"], refs, cap=2)[0] == 3
    print("publicity_controls.demo OK  (exact/1/2/length-mismatch distances and capping)")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
