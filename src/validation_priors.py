"""Measured false-positive fraction f of curated TCR-pMHC positives, from an independent re-assay.
# 2026-11-05  (added at revision round 1: reviewers 1.2, 3.2 asked for the epitope count and the
#              confidence interval behind f, and challenged f as circularly inferred from scores)

The score ceiling of src/validation_efficiency.py, AUC_max = 1 - f/2, is only as good as f. Until
now f entered the manuscript as a continuum over F_RANGE with the ~50% figure quoted as a property
of someone else's experiment, never as a measurement with an interval. Messemaker et al. (Immunity
2026) Table S7 closes that: it reports, per epitope, how many annotated VDJdb clonotypes an
independent functional screen could and could not reproduce.

Two readings are emitted, and they answer different objections:

1. POOLED, over all eight epitopes. This is the number reviewers 1.2 and 3.2 asked for -- f with an
   interval and a stated n -- and it is measured, not inferred from any model's score. It refutes
   the f = 10-20% counter-scenario directly: the best-curated database available is not in that
   range.

2. MOTIF-STRATIFIED, from the same re-assay (Shugay et al. 2026): records carrying an
   enrichment-derived convergence motif validated far better than motif-free ones. IMMREP25's
   positives sit at the repertoire background, i.e. in the motif-free stratum, so that stratum's
   rate -- not the pooled one -- is the relevant transfer. It is reported as a LOWER bound on f for
   IMMREP25, never as a point value, on the standing three-channel ordering argument of
   src/informativeness.py.

Nothing here is a measurement on IMMREP25, and the module asserts no value of f for it. What it
supplies is the anchor that the inequality is taken from.

Usage: uv run python src/validation_priors.py          # writes results/ + .dat + macros
       uv run python src/validation_priors.py --demo   # self-check only, no writes
"""
from __future__ import annotations
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ms_path                                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")
PRIORS = os.path.join(REPO, "dump", "messemaker", "validation_priors.tsv")

# Table S7 of Messemaker et al., Immunity 2026 (doi 10.1016/j.immuni.2026.09.007). Local only --
# dump/ is gitignored by repo policy; results/validation_priors.csv is the committed derivative.
# See SOURCES.md for how to obtain it.
COUNT_COLS = ["number_annotated_tcrs", "number_ambiguous_tcrs", "number_validating_tcrs",
              "number_non-validating_tcrs", "number_annotated_tcrs_wo_ambiguous",
              "test_set_count"]

# The motif stratification of the SAME re-assay, over two HLA-A*02:01 epitopes
# (Shugay et al. 2026, Nat Methods; already quoted in the Introduction as an odds ratio).
# (validating, total) for records matching an enrichment-derived motif, and for motif-free ones.
MOTIF = (174, 214)
MOTIF_FREE = (293, 773)

Z = 1.96


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because several per-epitope cells are small and
    the proportions run close to the boundary, where the Wald interval misbehaves.
    """
    if n == 0:
        raise ValueError("Wilson interval is undefined for n = 0")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return centre - half, centre + half


def read_priors(path: str = PRIORS) -> dict[str, dict[str, float]]:
    """Read Table S7, asserting its internal arithmetic rather than trusting the transcription.

    The invariants are the source table's own: every annotated clonotype is exactly one of
    ambiguous / validating / non-validating; the reported fraction's denominator EXCLUDES the
    ambiguous ones; and the rounded fraction in the file must agree with the counts. An ambiguous
    outcome is an outcome the assay could not call -- it is excluded, never folded into either
    class, which would silently move f.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        pep = row["Epitope"]
        if pep in out:
            raise ValueError("duplicate epitope in the priors table: %s" % pep)
        c = {k: int(row[k]) for k in COUNT_COLS}
        n, amb, val, non, wo = (c["number_annotated_tcrs"], c["number_ambiguous_tcrs"],
                                c["number_validating_tcrs"], c["number_non-validating_tcrs"],
                                c["number_annotated_tcrs_wo_ambiguous"])
        if min(c.values()) < 0 or n != amb + val + non or wo != val + non or wo == 0:
            raise ValueError("inconsistent counts for %s" % pep)
        if abs(float(row["Fraction_validating"]) - val / wo) > 5.1e-5:
            raise ValueError("rounded fraction disagrees with the counts for %s" % pep)
        lo, hi = wilson(val, wo)
        out[pep] = {"validating": val, "non_validating": non, "ambiguous": amb, "denom": wo,
                    "rate": val / wo, "f": 1.0 - val / wo,
                    "f_lo": 1.0 - hi, "f_hi": 1.0 - lo}
    return out


def pooled(priors: dict[str, dict[str, float]]) -> dict[str, float]:
    """Pooled validation rate over every epitope, and the f it implies."""
    val = int(sum(p["validating"] for p in priors.values()))
    den = int(sum(p["denom"] for p in priors.values()))
    lo, hi = wilson(val, den)
    return {"validating": val, "denom": den, "rate": val / den, "f": 1.0 - val / den,
            "f_lo": 1.0 - hi, "f_hi": 1.0 - lo, "n_epitopes": len(priors)}


def ceiling(f: float) -> float:
    """The label ceiling 1 - f/2, restated here so this module's outputs are self-contained.

    Identical to validation_efficiency.auc_ceiling(f, g=0); demo() asserts the two agree rather
    than letting a second expression for one quantity drift.
    """
    return 1.0 - f / 2.0


def _stratum(name: str, val: int, total: int) -> dict[str, float]:
    lo, hi = wilson(val, total)
    return {"stratum": name, "validating": val, "denom": total, "rate": val / total,
            "f": 1.0 - val / total, "f_lo": 1.0 - hi, "f_hi": 1.0 - lo,
            "ceiling": ceiling(1.0 - val / total),
            "ceiling_at_f_lo": ceiling(1.0 - hi), "ceiling_at_f_hi": ceiling(1.0 - lo)}


def run():
    priors = read_priors()
    pool = pooled(priors)
    motif = _stratum("motif", *MOTIF)
    free = _stratum("motif_free", *MOTIF_FREE)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)

    with open(os.path.join(RESULTS, "validation_priors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epitope", "validating", "non_validating", "ambiguous", "denom",
                    "validation_rate", "f", "f_lo", "f_hi", "ceiling"])
        for pep in sorted(priors):
            p = priors[pep]
            w.writerow([pep, int(p["validating"]), int(p["non_validating"]), int(p["ambiguous"]),
                        int(p["denom"]), "%.4f" % p["rate"], "%.4f" % p["f"],
                        "%.4f" % p["f_lo"], "%.4f" % p["f_hi"], "%.4f" % ceiling(p["f"])])
        w.writerow(["POOLED", pool["validating"], pool["denom"] - pool["validating"], "",
                    pool["denom"], "%.4f" % pool["rate"], "%.4f" % pool["f"],
                    "%.4f" % pool["f_lo"], "%.4f" % pool["f_hi"], "%.4f" % ceiling(pool["f"])])

    with open(os.path.join(RESULTS, "validation_strata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(motif))
        w.writeheader()
        for s in (motif, free):
            w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in s.items()})

    # gnuplot input: per-epitope f with its interval, ordered as the table is
    with open(os.path.join(ADAT, "validation_priors.dat"), "w") as fh:
        fh.write("# idx epitope f f_lo f_hi n\n")
        for i, pep in enumerate(sorted(priors), start=1):
            p = priors[pep]
            fh.write("%d %s %.4f %.4f %.4f %d\n"
                     % (i, pep, p["f"], p["f_lo"], p["f_hi"], int(p["denom"])))

    # Supplementary table: the per-epitope anchor behind f, which reviewer 1.2 asked to see.
    rows = sorted(priors, key=lambda k: priors[k]["f"])
    with open(ms_path("tables", "validation_priors_table.tex"), "w") as fh:
        fh.write("%% GENERATED by src/validation_priors.py -- do not edit.\n"
                 "%% Regenerate: python src/validation_priors.py "
                 "(from the 2026-immrep25-audit repo).\n")
        fh.write("\\begin{tabular}{lrrrrl}\n\\toprule\n")
        fh.write("epitope & validated & not validated & ambiguous & $f$ & 95\\% CI \\\\\n")
        fh.write("\\midrule\n")
        for pep in rows:
            p = priors[pep]
            fh.write("\\texttt{%s} & %d & %d & %d & %.3f & %.3f--%.3f \\\\\n"
                     % (pep, int(p["validating"]), int(p["non_validating"]),
                        int(p["ambiguous"]), p["f"], p["f_lo"], p["f_hi"]))
        fh.write("\\midrule\n")
        fh.write("pooled & %d & %d & & %.3f & %.3f--%.3f \\\\\n"
                 % (pool["validating"], pool["denom"] - pool["validating"],
                    pool["f"], pool["f_lo"], pool["f_hi"]))
        fh.write("\\bottomrule\n\\end{tabular}\n")

    fs = [p["f"] for p in priors.values()]
    macros = {
        # --- pooled: the anchor reviewers 1.2 and 3.2 asked for -------------------------------
        "vpNepi": "%d" % pool["n_epitopes"],
        "vpNclono": "{:,}".format(pool["denom"]).replace(",", "\\,"),
        "vpNval": "{:,}".format(pool["validating"]).replace(",", "\\,"),
        "vpRate": "%.4f" % pool["rate"],
        "vpF": "%.4f" % pool["f"],
        "vpFlo": "%.4f" % pool["f_lo"],
        "vpFhi": "%.4f" % pool["f_hi"],
        "vpFpct": "%.1f" % (100 * pool["f"]),
        "vpFpctLo": "%.1f" % (100 * pool["f_lo"]),
        "vpFpctHi": "%.1f" % (100 * pool["f_hi"]),
        "vpCeil": "%.4f" % ceiling(pool["f"]),
        # spread across epitopes: the headline is pooled, but f is not homogeneous
        "vpFmin": "%.4f" % min(fs),
        "vpFmax": "%.4f" % max(fs),
        "vpPepMin": min(priors, key=lambda k: priors[k]["f"]),
        "vpPepMax": max(priors, key=lambda k: priors[k]["f"]),
        # --- motif stratification: the transfer to IMMREP25's region of sequence space --------
        "vpMotifVal": "%d" % MOTIF[0],
        "vpMotifN": "%d" % MOTIF[1],
        "vpMotifF": "%.4f" % motif["f"],
        "vpMotifCeil": "%.4f" % motif["ceiling"],
        "vpFreeVal": "%d" % MOTIF_FREE[0],
        "vpFreeN": "%d" % MOTIF_FREE[1],
        "vpFreeF": "%.4f" % free["f"],
        "vpFreeFlo": "%.4f" % free["f_lo"],
        "vpFreeFhi": "%.4f" % free["f_hi"],
        "vpFreeFpct": "%.1f" % (100 * free["f"]),
        "vpFreeCeil": "%.4f" % free["ceiling"],
        "vpFreeCeilLo": "%.4f" % free["ceiling_at_f_hi"],
        "vpFreeCeilHi": "%.4f" % free["ceiling_at_f_lo"],
    }
    with open(os.path.join(ADAT, "validation_priors_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("pooled over %d epitopes (%d clonotypes, ambiguous excluded): validation rate %.4f, "
          "f = %.4f (95%% CI %.4f-%.4f), ceiling %.4f"
          % (pool["n_epitopes"], pool["denom"], pool["rate"], pool["f"], pool["f_lo"],
             pool["f_hi"], ceiling(pool["f"])))
    print("per-epitope f spans %.4f (%s) to %.4f (%s)"
          % (min(fs), macros["vpPepMin"], max(fs), macros["vpPepMax"]))
    print("motif-free stratum: f = %.4f (95%% CI %.4f-%.4f), ceiling %.4f -- this is the stratum "
          "IMMREP25's positives occupy, and it enters as a LOWER bound, never a point value"
          % (free["f"], free["f_lo"], free["f_hi"], free["ceiling"]))


def demo():
    """Load-bearing self-checks."""
    priors = read_priors()
    assert len(priors) == 8, "expected 8 epitopes in Table S7, got %d" % len(priors)
    pool = pooled(priors)
    assert pool["denom"] == sum(p["denom"] for p in priors.values())
    # the interval must bracket the point estimate, and lie inside [0,1]
    assert pool["f_lo"] < pool["f"] < pool["f_hi"], "pooled interval does not bracket f"
    assert 0.0 <= pool["f_lo"] and pool["f_hi"] <= 1.0

    # f is NOT in the 10-20% range the reviewers posed as the optimistic counter-scenario; this
    # assert is the claim the manuscript makes, so it fails loudly if the table ever changes.
    assert pool["f_lo"] > 0.20, "pooled f CI reaches into the 10-20%% range: %.4f" % pool["f_lo"]

    # the motif-free stratum must sit ABOVE the pooled rate -- that ordering is what licenses
    # reading it onto a cohort at the repertoire background
    free = _stratum("motif_free", *MOTIF_FREE)
    motif = _stratum("motif", *MOTIF)
    assert free["f"] > pool["f"] > motif["f"], "motif stratification does not order as expected"

    # and it must exceed veFatCeil 0.545, the crossover past which the shortfall is unresolvable
    assert free["f_lo"] > 0.545, ("the motif-free interval no longer clears the resolution "
                                  "crossover: %.4f" % free["f_lo"])

    # this module's ceiling must agree with the one validation_efficiency.py derives
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validation_efficiency import auc_ceiling
    for f in (0.0, 0.1, 0.25, 0.4436, 0.621, 0.8):
        assert abs(ceiling(f) - auc_ceiling(f, 0.0)) < 1e-12, "two ceilings disagree at f=%.4f" % f

    # Wilson must reduce to the point estimate as n grows, and bracket a known value
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    print("demo OK: %d epitopes, pooled f = %.4f (95%% CI %.4f-%.4f); motif-free f = %.4f "
          "(%.4f-%.4f)" % (len(priors), pool["f"], pool["f_lo"], pool["f_hi"],
                           free["f"], free["f_lo"], free["f_hi"]))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
