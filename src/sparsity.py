"""How much of a positive set is LEARNABLE at the depth it was sampled, and what that caps.
# 2026-11-05  (added at revision round 1, for reviewer 1.1's request that the ceiling formula
#              carry a sparsity term, and reviewers 3.1/3.2 on what explains the 0.60 plateau)

The label ceiling AUC_max = 1 - f/2 of src/validation_efficiency.py assumes every correctly
labelled positive is RECOVERABLE. It need not be. Epitope-specific recognition is carried by
convergent groups of receptors; a positive whose group is sampled once in the benchmark shares no
within-epitope structure with anything, so no probe reading convergence -- and no model that must
learn the rule from this benchmark -- can place it above background. Sampling k = 1 from a group
carries no more information than sampling none.

Define coverage c = the fraction of a cohort's positives that have at least one same-epitope
neighbour at Hamming <= 1, i.e. that sit in a within-epitope group of size >= 2. Then two bounds
are reported side by side, and they answer different questions:

    label ceiling      1 - f/2                  what a predictor with external knowledge could
                                                reach, limited only by the labels;
    learnable ceiling  1 - (1 - c(1-f))/2       what this benchmark can itself teach and verify,
                                                limited by labels AND by sampling depth.

Both are oracle bounds. The second is deliberately NOT a claim that no method can exceed it --
IMMREP25's reported best does exceed it, and that is the finding: a score above what the
benchmark's own data can support is evidence that the leading entries import knowledge from
elsewhere and are not adapting to this benchmark at all. A benchmark whose learnable ceiling sits
below its own leaderboard cannot be ranking what it claims to rank.

Coverage depends on sampling depth, so every cohort is sampled at approximately 50 receptors
per epitope (a cap of PER_EPI = 50) before counting. Comparing a cohort sampled at 600 per epitope with
one sampled at 50 would measure the sampling, not the biology. The subsample is repeated over
N_DRAW seeds and the interval is the spread across draws. Where every qualifying epitope has
no more than PER_EPI receptors, subsampling retains all receptors and the interval collapses
to a point.

The inverse reading is the more interpretable one and is emitted too: solving the same sampling
model for N gives the effective number of distinct specificity solutions per epitope that the
observed convergence implies. A benchmark sampling 50 receptors from ~1000 solutions sees almost
every receptor as a singleton, which is the sparsity the argument is about.

Usage: uv run python src/sparsity.py          # writes results/ + .dat + macros
       uv run python src/sparsity.py --demo   # self-checks only, no writes
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts, HIERARCHY, COHORT_META          # noqa: E402
from homology_graph import build_graph                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")

MIN_N = 30        # epitopes with fewer records are not scored, as everywhere else in this repo
PER_EPI = 50      # sampling cap, approximately matching IMMREP25's per-epitope depth
N_DRAW = 200      # subsample repeats; the reported interval is the spread over these
CHAINS = ("A", "B")
SEED = 0


def coverage_once(sub) -> tuple[float, int, int, float]:
    """(coverage, n_receptors, n_groups, mean_group_size) for one subsample of one cohort/chain.

    A "group" is a connected component of the WITHIN-epitope subgraph: receptors of the same
    epitope joined by Hamming <= 1. Singletons count as groups of one, so n_groups includes them
    and mean group size is n_receptors / n_groups. Coverage is the share of receptors in a group
    of size >= 2 -- the share on which any convergence-reading method has anything to work with.
    """
    G = build_graph(sub[["cdr3", "epitope"]])
    # keep only within-epitope edges: a cross-epitope neighbour is not evidence about either label
    G.remove_edges_from([(u, v) for u, v in G.edges
                         if G.nodes[u]["epitope"] != G.nodes[v]["epitope"]])
    n = G.number_of_nodes()
    if n == 0:
        return float("nan"), 0, 0, float("nan")
    import networkx as nx
    comps = list(nx.connected_components(G))
    covered = sum(len(c) for c in comps if len(c) >= 2)
    return covered / n, n, len(comps), n / len(comps)


def analytic_coverage(n_groups: int, m: int = PER_EPI) -> float:
    """Coverage expected if m receptors are drawn uniformly from n_groups equally likely groups.

    P(a drawn receptor's group is hit by at least one of the other m-1 draws)
      = 1 - (1 - 1/N)^(m-1).
    """
    if n_groups <= 0:
        return float("nan")
    return 1.0 - (1.0 - 1.0 / n_groups) ** (m - 1)


def implied_solutions(c: float, m: int = PER_EPI) -> float:
    """Invert coverage to the effective number of specificity solutions N per epitope.

    This is the quantity the sparsity argument is really about: an epitope is recognised by some
    number of distinct convergent solutions, and a benchmark that samples m receptors sees a
    same-group pair only when it draws the same solution twice. Solving
    1 - (1 - 1/N)^(m-1) = c for N gives the N consistent with the convergence actually observed,
    at the depth actually sampled.

    It is an EFFECTIVE count under a uniform-group model, not a census: real solutions differ in
    frequency, and unequal frequencies raise coverage at fixed N, so this reads as a lower bound
    on how many solutions a benchmark of this depth would have to cover. Returns inf at c = 0,
    where no finite N is implied.
    """
    if not (0.0 < c < 1.0):
        return float("inf") if c <= 0.0 else float("nan")
    return 1.0 / (1.0 - (1.0 - c) ** (1.0 / (m - 1)))


def learnable_ceiling(c: float, f: float) -> float:
    """Ceiling for a predictor that can only use what this benchmark itself carries.

    A positive counts as usable only if it is a true binder (probability 1-f) AND sits in a
    sampled group (probability c). The usable fraction is w = c(1-f); the remaining 1-w of the
    positive class is indistinguishable from the negatives, exactly as the mislabelled fraction
    is in the 1 - f/2 argument, so each contributes half and

        AUC_max_learnable = w + (1-w)/2 = 1 - (1-w)/2.

    At c = 1 this reduces to the label ceiling 1 - f/2, which demo() asserts.
    """
    w = c * (1.0 - f)
    return 1.0 - (1.0 - w) / 2.0


def per_cohort(coh: dict, chain: str, n_draw: int = N_DRAW,
               seed: int = SEED) -> dict[str, dict[str, float]]:
    """Coverage and group structure per cohort using a common per-epitope sampling cap."""
    out: dict[str, dict[str, float]] = {}
    for name in HIERARCHY:
        if name not in coh:
            continue
        lc = coh[name]["long"]
        lc = lc[lc.chain == chain]
        vc = lc.epitope.value_counts()
        eps = vc[vc >= MIN_N].index.to_numpy()
        if eps.size == 0:
            continue
        df = lc[lc.epitope.isin(eps)].reset_index(drop=True)
        cs, ns, gs, ms = [], [], [], []
        for d in range(n_draw):
            sh = df.sample(frac=1.0, random_state=seed + d)
            sub = sh[sh.groupby("epitope").cumcount() < PER_EPI].reset_index(drop=True)
            c, n, g, msz = coverage_once(sub)
            cs.append(c); ns.append(n); gs.append(g); ms.append(msz)
        cs = np.asarray(cs, float)
        out[name] = {
            "n_epitopes": int(eps.size),
            "n_receptors": float(np.mean(ns)),
            "coverage": float(np.mean(cs)),
            "coverage_lo": float(np.percentile(cs, 2.5)),
            "coverage_hi": float(np.percentile(cs, 97.5)),
            "n_groups": float(np.mean(gs)),
            "groups_per_epitope": float(np.mean(gs)) / eps.size,
            "mean_group_size": float(np.mean(ms)),
            "implied_solutions": implied_solutions(float(np.mean(cs))),
        }
    return out


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validation_priors import read_priors, pooled, _stratum, MOTIF_FREE
    from validation_efficiency import auc_ceiling, AUC_BEST

    pool = pooled(read_priors())
    free = _stratum("motif_free", *MOTIF_FREE)
    f_pool, f_free = pool["f"], free["f"]

    # include_olga=True brings in the synthetic and real controls (pairSEQ mock,
    # MLR expanded, AIRR random, OLGA random); they are the background this
    # coverage has to be read against, so they are not optional here.
    coh = build_cohorts(include_olga=True)
    tables = {ch: per_cohort(coh, ch) for ch in CHAINS}

    with open(os.path.join(RESULTS, "sparsity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "label", "chain", "n_epitopes", "n_receptors", "n_groups",
                    "groups_per_epitope", "mean_group_size", "coverage", "coverage_lo",
                    "coverage_hi", "implied_solutions", "learnable_ceiling_f_pooled",
                    "learnable_ceiling_f_motif_free", "label_ceiling_f_pooled"])
        for ch in CHAINS:
            for name, r in tables[ch].items():
                w.writerow([name, COHORT_META[name]["label"], ch, r["n_epitopes"],
                            "%.1f" % r["n_receptors"], "%.1f" % r["n_groups"],
                            "%.2f" % r["groups_per_epitope"], "%.3f" % r["mean_group_size"],
                            "%.4f" % r["coverage"], "%.4f" % r["coverage_lo"],
                            "%.4f" % r["coverage_hi"], "%.1f" % r["implied_solutions"],
                            "%.4f" % learnable_ceiling(r["coverage"], f_pool),
                            "%.4f" % learnable_ceiling(r["coverage"], f_free),
                            "%.4f" % auc_ceiling(f_pool)])

    # gnuplot input: coverage per cohort, both chains, in the standard hierarchy order
    with open(os.path.join(ADAT, "sparsity.dat"), "w") as fh:
        fh.write("# idx cohort coverage_A lo_A hi_A coverage_B lo_B hi_B label\n")
        for i, name in enumerate([n for n in HIERARCHY if n in tables["B"]], start=1):
            a = tables["A"].get(name)
            b = tables["B"][name]
            fh.write("%d %s %s %s %s %.4f %.4f %.4f \"%s\"\n"
                     % (i, name,
                        "%.4f" % a["coverage"] if a else "NaN",
                        "%.4f" % a["coverage_lo"] if a else "NaN",
                        "%.4f" % a["coverage_hi"] if a else "NaN",
                        b["coverage"], b["coverage_lo"], b["coverage_hi"],
                        COHORT_META[name]["label"]))

    imm = tables["B"]["immrep25_pos"]
    ii = tables["B"].get("immrep22_true")
    hq = tables["B"].get("vdjdb_hq")
    mock = tables["B"].get("pairseq_mock")
    imm_a = tables["A"]["immrep25_pos"]

    lc_free = learnable_ceiling(imm["coverage"], f_free)
    lc_pool = learnable_ceiling(imm["coverage"], f_pool)
    macros = {
        "spPerEpi": "%d" % PER_EPI,
        "spNdraw": "%d" % N_DRAW,
        # --- coverage, the measured sparsity -------------------------------------------------
        "spCovImmB": "%.3f" % imm["coverage"],
        "spCovImmBLo": "%.3f" % imm["coverage_lo"],
        "spCovImmBHi": "%.3f" % imm["coverage_hi"],
        "spCovImmBPct": "%.1f" % (100 * imm["coverage"]),
        "spCovImmA": "%.3f" % imm_a["coverage"],
        "spCovImmAPct": "%.1f" % (100 * imm_a["coverage"]),
        "spCovIiB": "%.3f" % ii["coverage"],
        "spCovIiBPct": "%.1f" % (100 * ii["coverage"]),
        "spCovHqB": "%.3f" % hq["coverage"],
        "spCovHqBPct": "%.1f" % (100 * hq["coverage"]),
        "spCovPsB": "%.3f" % mock["coverage"],
        "spCovPsBPct": "%.1f" % (100 * mock["coverage"]),
        "spCovRatioIi": "%.1f" % (ii["coverage"] / imm["coverage"]),
        # --- group structure: the N and k of the sparsity argument ----------------------------
        "spGroupsImmB": "%.1f" % imm["groups_per_epitope"],
        "spSizeImmB": "%.2f" % imm["mean_group_size"],
        "spGroupsIiB": "%.1f" % ii["groups_per_epitope"],
        "spSizeIiB": "%.2f" % ii["mean_group_size"],
        # the effective number of distinct specificity solutions per epitope that the
        # observed convergence implies, against the PER_EPI actually sampled from them
        "spNimpliedImmB": "%.0f" % imm["implied_solutions"],
        "spNimpliedIiB": "%.0f" % ii["implied_solutions"],
        "spNimpliedTtB": "%.0f" % tables["B"]["tcrvdb_true"]["implied_solutions"],
        "spNimpliedHqB": "%.0f" % hq["implied_solutions"],
        # --- the two ceilings, side by side ---------------------------------------------------
        "spLearnImmFree": "%.3f" % lc_free,
        "spLearnImmPool": "%.3f" % lc_pool,
        "spLearnIiPool": "%.3f" % learnable_ceiling(ii["coverage"], f_pool),
        "spLabelCeilPool": "%.3f" % auc_ceiling(f_pool),
        # the gap that carries the argument: the leaderboard best EXCEEDS the learnable ceiling
        "spBestOverLearn": "%.3f" % (AUC_BEST - lc_free),
        "spBestOverLearnPool": "%.3f" % (AUC_BEST - lc_pool),
    }
    with open(os.path.join(ADAT, "sparsity_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("coverage at approximately %d receptors/epitope, %d draws (TCRbeta):" %
          (PER_EPI, N_DRAW))
    for name in HIERARCHY:
        if name in tables["B"]:
            r = tables["B"][name]
            print("  %-14s c=%.3f (%.3f-%.3f)  implied solutions/epitope %7.0f  "
                  "mean group size %.2f  (%d epitopes)"
                  % (COHORT_META[name]["label"], r["coverage"], r["coverage_lo"],
                     r["coverage_hi"], r["implied_solutions"], r["mean_group_size"],
                     r["n_epitopes"]))
    print("IMMREP25 learnable ceiling %.3f at the motif-free f=%.3f (%.3f at the pooled f=%.3f), "
          "against a label ceiling of %.3f and a reported best of %.2f -- the best entry exceeds "
          "what this benchmark's own data can support by %.3f"
          % (lc_free, f_free, lc_pool, f_pool, auc_ceiling(f_pool), AUC_BEST, AUC_BEST - lc_free))


def demo():
    # the learnable ceiling must reduce to the label ceiling at full coverage
    from validation_efficiency import auc_ceiling
    for f in (0.0, 0.1, 0.4436, 0.621, 0.8):
        assert abs(learnable_ceiling(1.0, f) - auc_ceiling(f, 0.0)) < 1e-12, \
            "learnable ceiling does not reduce to 1-f/2 at c=1 (f=%.4f)" % f
    # and be monotone in both arguments, and never below chance
    assert learnable_ceiling(0.5, 0.3) < learnable_ceiling(0.9, 0.3)
    assert learnable_ceiling(0.5, 0.6) < learnable_ceiling(0.5, 0.3)
    assert abs(learnable_ceiling(0.0, 0.0) - 0.5) < 1e-12, "zero coverage must give chance"

    # the inversion must round-trip against the forward model it inverts
    for n in (50, 152, 425, 975, 5000):
        assert abs(implied_solutions(analytic_coverage(n)) - n) < 1e-6, n
    assert implied_solutions(0.0) == float("inf")

    # the analytic companion must be monotone decreasing in the number of groups and bracket [0,1]
    vals = [analytic_coverage(n) for n in (10, 50, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert 0.0 < vals[-1] < vals[0] < 1.0

    # coverage on a hand-built case with a known answer: per epitope, one Hamming-1 pair and
    # one isolated receptor, so coverage is exactly 2/3 and there are 2 groups of the 3 receptors.
    import pandas as pd
    rows = []
    for e, (a, b, lone) in zip("XYZ", [
            ("CASSLGQAYEQYF", "CASSLGQAYEQWF", "CATTRDSNQPQHF"),
            ("CASSPRDRGNTIY", "CASSPRDRGNTIW", "CAWSVGDEKLFFA"),
            ("CASSLAPGATNEK", "CASSLAPGATNEW", "CAISEYGNKLVFG")]):
        rows += [(a, e), (b, e), (lone, e)]
    df = pd.DataFrame(rows, columns=["cdr3", "epitope"])
    assert df.cdr3.is_unique and len(df) == 9
    c, n, g, msz = coverage_once(df)
    assert n == 9 and g == 6, "expected 9 receptors in 6 groups, got n=%d g=%d" % (n, g)
    assert abs(c - 2.0 / 3.0) < 1e-12, "hand-built coverage should be exactly 2/3, got %.4f" % c
    assert abs(msz - 1.5) < 1e-12, msz

    # a cross-epitope neighbour must NOT count: move the lone receptors onto one shared sequence
    rows2 = [(a, e) for a, e in rows if not a.startswith("CAT") and not a.startswith("CAW")
             and not a.startswith("CAI")]
    shared = [("CATTRDSNQPQHF", "X"), ("CATTRDSNQPQHW", "Y")]
    c2, n2, g2, _ = coverage_once(pd.DataFrame(rows2 + shared,
                                               columns=["cdr3", "epitope"]))
    assert abs(c2 - 6.0 / 8.0) < 1e-12, ("a Hamming-1 pair spanning two epitopes must not count "
                                         "as covered; got %.4f" % c2)

    print("demo OK: learnable ceiling reduces to 1-f/2 at c=1; hand case c=%.3f over n=%d "
          "receptors in %d groups, cross-epitope edges correctly excluded (c=%.3f)"
          % (c, n, g, c2))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
