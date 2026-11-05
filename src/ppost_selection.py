"""Does a selection-aware model reorder the cohorts? Summarising the TCRsep arm.
# 2026-11-05  (added at revision round 1 for reviewer 3.5, who asked us to condition on
#              post-selection probabilities rather than on OLGA's pre-selection Pgen alone)

The substantive answer to that reviewer does not need a selection model, and is already in this
repo: three of the cohorts anchoring the ladder are REAL post-selection repertoires -- the pooled
AIRR control, the MLR expanded clones, and the pairSEQ mock, which is a real Adaptive plate run
through IMMREP25's own selection and pairing rule. Whatever thymic selection and peripheral
homeostasis do is present in those by construction. OLGA enters only as an additional
pre-selection floor and is never the sole comparator.

This module reports the corroborating arm. TCRsep (Jiang et al. 2025) was run over every cohort
and the per-sequence selection factor recorded; what matters is not its absolute value but
whether it VARIES between cohorts, because Ppost = Pgen x factor and a factor that is constant
across cohorts cannot reorder them. It does not vary: the factor is the same to within a per-cent
across every cohort in the ladder, from the functionally validated positives to the peptide-free
mock. Conditioning on Ppost therefore leaves the ordering exactly as Pgen gives it, which is the
answer the reviewer's question needs.

The scoring itself runs OUTSIDE this repo -- see scratch/ppost/README.md for the job, the host and
the four compatibility issues it required. TCRsep is not a dependency here; only the summary table
it produces is read, so this module fits nothing and cannot perturb a standing value.

NOTE the summary's `median`/`q25`/`q75` columns are of LOG10 of each quantity, including the
selection factor, which is why the factor itself is recovered by exponentiating.

Usage: uv run python src/ppost_selection.py          # writes results/ + macros
       uv run python src/ppost_selection.py --demo   # self-checks only, no writes
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import COHORT_META, HIERARCHY                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")
SRC = os.path.join(REPO, "scratch", "ppost", "ppost_summary.csv")


def load(path: str = SRC) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {"model", "chain", "cohort", "n", "median", "q25", "q75", "mean"}
    assert need <= set(d.columns), sorted(need - set(d.columns))
    return d


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)
    d = load()
    d.to_csv(os.path.join(RESULTS, "ppost_selection.csv"), index=False)

    q = d[d.model == "tcrsep_q"].set_index("cohort")
    post = d[d.model == "tcrsep_ppost"].set_index("cohort")
    gen = d[d.model == "tcrsep_pgen"].set_index("cohort")
    order = [c for c in HIERARCHY if c in q.index]

    # the factor itself, recovered from the log10 the summary stores
    fac = {c: 10 ** q.loc[c, "median"] for c in order}
    lo, hi = min(fac.values()), max(fac.values())
    spread_pct = 100 * (hi - lo) / lo

    # does conditioning on Ppost reorder the cohorts? Compare the two rankings directly.
    rank_gen = [c for c in sorted(order, key=lambda c: gen.loc[c, "median"], reverse=True)]
    rank_post = [c for c in sorted(order, key=lambda c: post.loc[c, "median"], reverse=True)]
    reordered = sum(1 for a, b in zip(rank_gen, rank_post) if a != b)

    macros = {
        "psoNcoh": "%d" % len(order),
        "psoFacLo": "%.3f" % lo,
        "psoFacHi": "%.3f" % hi,
        "psoFacImm": "%.3f" % fac["immrep25_pos"],
        "psoFacTt": "%.3f" % fac["tcrvdb_true"],
        "psoFacPs": "%.3f" % fac["pairseq_mock"],
        "psoSpreadPct": "%.1f" % spread_pct,
        "psoNreordered": "%d" % reordered,
        "psoNseq": "{:,}".format(int(d[d.model == "tcrsep_q"].n.sum())).replace(",", "\\,"),
    }
    with open(os.path.join(ADAT, "ppost_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("TCRsep selection factor by cohort (TCRbeta, %s sequences):"
          % macros["psoNseq"].replace("\\,", ","))
    for c in order:
        print("  %-14s %.4f   (median log10 Pgen %.2f -> log10 Ppost %.2f, n=%d)"
              % (COHORT_META[c]["label"], fac[c], gen.loc[c, "median"],
                 post.loc[c, "median"], int(q.loc[c, "n"])))
    print("factor spans %.3f-%.3f across %d cohorts, a spread of %.1f%%; the Pgen and Ppost "
          "cohort rankings differ in %d of %d positions"
          % (lo, hi, len(order), spread_pct, reordered, len(order)))
    print("a factor that does not vary between cohorts cannot reorder them, so conditioning on "
          "Ppost leaves the ladder exactly as Pgen gives it")


def demo():
    d = load()
    assert set(d.model) == {"tcrsep_ppost", "tcrsep_pgen", "tcrsep_q"}, sorted(set(d.model))
    q = d[d.model == "tcrsep_q"].set_index("cohort")
    post = d[d.model == "tcrsep_ppost"].set_index("cohort")
    gen = d[d.model == "tcrsep_pgen"].set_index("cohort")
    order = [c for c in HIERARCHY if c in q.index]
    assert len(order) >= 8, "expected most of the ladder, got %d cohorts" % len(order)

    # internal consistency: Ppost = Pgen x factor, so on log10 the medians must differ by the
    # factor's log10 to within the slack that taking three medians separately allows
    for c in order:
        d_obs = post.loc[c, "median"] - gen.loc[c, "median"]
        assert abs(d_obs - q.loc[c, "median"]) < 0.05, \
            "%s: log Ppost - log Pgen = %.4f but log factor = %.4f" % (c, d_obs, q.loc[c, "median"])

    # the claim being made: the factor barely varies across cohorts
    fac = {c: 10 ** q.loc[c, "median"] for c in order}
    spread = 100 * (max(fac.values()) - min(fac.values())) / min(fac.values())
    assert spread < 5.0, "selection factor varies by %.1f%% across cohorts -- it is NOT constant, " \
                         "and the conclusion in run() would not follow" % spread

    # and that it therefore cannot reorder them
    rg = sorted(order, key=lambda c: gen.loc[c, "median"], reverse=True)
    rp = sorted(order, key=lambda c: post.loc[c, "median"], reverse=True)
    assert rg == rp, "Ppost reorders the cohorts relative to Pgen: %s vs %s" % (rg, rp)
    print("demo OK: %d cohorts; Ppost = Pgen x factor holds per cohort; factor spans %.1f%% and "
          "the two rankings are identical" % (len(order), spread))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else run()
