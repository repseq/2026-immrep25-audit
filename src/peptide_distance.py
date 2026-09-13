#!/usr/bin/env python
# 2026-09-13  How dissimilar are IMMREP25's own peptides from each other?
#
# The benchmark forms its 9,000 negatives by re-pairing each receptor with the other nine
# peptides OF ITS OWN MHC, so a labelled negative is a false negative only if that receptor
# genuinely binds two peptides of the same ten-peptide panel. Every negative receptor was
# detected as a binder of some other peptide in the panel, so assay insensitivity -- the usual
# route to an undetected binder -- cannot produce one; the only route is real cross-reactivity
# within the panel.
#
# TCR cross-reactivity is not deniable in general (a single receptor is usually argued to
# recognise on the order of 1e6 peptides), but degeneracy concentrates on MUTUALLY SIMILAR
# peptides. So the quantity that bounds the within-panel false-negative rate is how similar
# the panel's peptides are to each other -- measured here, within allele, on the benchmark's
# own twenty peptides.
#
# This is descriptive: no null model, no permutation. It exists because the published distance
# figure for IMMREP25 (>=4 Levenshtein, no shared substring >5 residues; Garcia Noceda et al.)
# is distance from PUBLISHED peptides, i.e. from the training corpus -- not distance BETWEEN
# the twenty, which is what the cross-reactivity argument needs.
#
# Output is tidy: one row per peptide pair, one column per distance; `kind` separates the
# within-allele pairs (the ones the benchmark re-pairs across) from the cross-allele pairs,
# which it never forms.
#
# Run: python src/peptide_distance.py      (--demo for the self-check)
from __future__ import annotations
import os
import sys
from itertools import combinations

import polars as pl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")


def lev(a: str, b: str) -> int:
    """Levenshtein edit distance (unit cost). Iterative single-row DP."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                   # deletion
                           cur[j - 1] + 1,                # insertion
                           prev[j - 1] + (ca != cb)))     # substitution
        prev = cur
    return prev[-1]


def hamming(a: str, b: str) -> int:
    """Position-wise mismatches; the panel is all 9-mers so this is always defined."""
    assert len(a) == len(b), "hamming needs equal lengths"
    return sum(x != y for x, y in zip(a, b))


def lcsubstr(a: str, b: str) -> int:
    """Length of the longest COMMON CONTIGUOUS substring (not subsequence)."""
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            run = prev[j - 1] + 1 if ca == cb else 0
            cur.append(run)
            best = max(best, run)
        prev = cur
    return best


def load_panel() -> pl.DataFrame:
    """The benchmark's 20 positive peptides with their restricting allele."""
    return (pl.read_csv(DATA, separator="\t")
            .filter(pl.col("label") == 1)
            .select("hla", "peptide")
            .unique()
            .sort("hla", "peptide"))


def pair_table(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per peptide pair; `kind` distinguishes within-allele from cross-allele."""
    by_allele = {r["hla"]: r["peptide"] for r in
                 panel.group_by("hla").agg(pl.col("peptide")).sort("hla").to_dicts()}
    rows = []
    for hla, peps in by_allele.items():
        for p, q in combinations(sorted(peps), 2):
            rows.append(dict(kind="within", hla=hla, pep1=p, pep2=q))
    alleles = sorted(by_allele)
    for i, a in enumerate(alleles):
        for b in alleles[i + 1:]:
            for p in sorted(by_allele[a]):
                for q in sorted(by_allele[b]):
                    rows.append(dict(kind="cross", hla=None, pep1=p, pep2=q))
    return (pl.DataFrame(rows, schema={"kind": pl.Utf8, "hla": pl.Utf8,
                                       "pep1": pl.Utf8, "pep2": pl.Utf8})
            .with_columns(
                length=pl.col("pep1").str.len_chars(),
                lev=pl.struct("pep1", "pep2").map_elements(
                    lambda s: lev(s["pep1"], s["pep2"]), return_dtype=pl.Int64),
                hamming=pl.struct("pep1", "pep2").map_elements(
                    lambda s: hamming(s["pep1"], s["pep2"]), return_dtype=pl.Int64),
                lcsubstr=pl.struct("pep1", "pep2").map_elements(
                    lambda s: lcsubstr(s["pep1"], s["pep2"]), return_dtype=pl.Int64)))


def summarise(t: pl.DataFrame) -> pl.DataFrame:
    """Tidy summary: one row per (kind, allele, metric)."""
    return (t.unpivot(index=["kind", "hla"], on=["lev", "hamming", "lcsubstr"],
                      variable_name="metric", value_name="value")
            .group_by("kind", "hla", "metric")
            .agg(n_pairs=pl.len(), min=pl.col("value").min(),
                 median=pl.col("value").median(), max=pl.col("value").max())
            .sort("kind", "hla", "metric"))


def main():
    t = pair_table(load_panel())
    t.write_csv(os.path.join(RESULTS, "peptide_distance.csv"))
    s = summarise(t)
    s.write_csv(os.path.join(RESULTS, "peptide_distance_summary.csv"))

    print("=== peptide-pair distances, one row per (pair kind, allele, metric) ===")
    print("lev = Levenshtein edit distance; hamming = position-wise mismatches; "
          "lcsubstr = longest shared contiguous substring, residues")
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(s)

    within = t.filter(pl.col("kind") == "within")
    n_pairs = within.height
    lev_min = int(within["lev"].min())
    ham_min = int(within["hamming"].min())
    lcs_max = int(within["lcsubstr"].max())
    n_lcs_gt5 = int(within.filter(pl.col("lcsubstr") > 5).height)
    n_lev_le3 = int(within.filter(pl.col("lev") <= 3).height)
    print("\nwithin-allele pairs (the pairs the benchmark re-pairs across): n=%d over %d "
          "peptides, all %d-mers" % (n_pairs, load_panel().height, int(within["length"][0])))
    print("  minimum Levenshtein distance %d edits; median %.0f" % (lev_min, within["lev"].median()))
    print("  minimum position-wise mismatch count %d of 9; median %.0f"
          % (ham_min, within["hamming"].median()))
    print("  longest shared contiguous substring %d residues; median %.0f"
          % (lcs_max, within["lcsubstr"].median()))
    print("  pairs sharing a substring longer than 5 residues: %d of %d" % (n_lcs_gt5, n_pairs))
    print("  pairs within 3 edits of each other: %d of %d" % (n_lev_le3, n_pairs))

    macros = {
        "pepNpairs": "%d" % n_pairs,
        "pepLevMin": "%d" % lev_min,
        "pepLevMed": "%.0f" % within["lev"].median(),
        "pepHamMin": "%d" % ham_min,
        "pepHamMed": "%.0f" % within["hamming"].median(),
        "pepLcsMax": "%d" % lcs_max,
        "pepLcsMed": "%.0f" % within["lcsubstr"].median(),
        "pepNlcsGtFive": "%d" % n_lcs_gt5,
        "pepNlevLeThree": "%d" % n_lev_le3,
        "pepLen": "%d" % int(within["length"][0]),
    }
    with open(os.path.join(ADAT, "peptide_distance_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/peptide_distance.csv, results/peptide_distance_summary.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "peptide_distance_macros.tex"),
                                    len(macros)))


def demo():
    """Self-check: the three distances on hand-worked cases, then the panel's shape,
    symmetry, and the equal-length inequality lev <= hamming."""
    assert lev("ABC", "ABC") == 0
    assert lev("ABC", "ABD") == 1                  # one substitution
    assert lev("AB", "ABC") == 1                   # one insertion
    assert lev("KITTEN", "SITTING") == 3           # the textbook case
    assert hamming("ABCD", "ABCE") == 1
    assert lcsubstr("ABCDE", "XBCDY") == 3         # "BCD"
    assert lcsubstr("ABCDE", "EDCBA") == 1         # no run longer than one residue
    assert lcsubstr("ABC", "XYZ") == 0
    panel = load_panel()
    assert panel.height == 20, "expected 20 peptides"
    assert set(panel["peptide"].str.len_chars().to_list()) == {9}, "panel is not all 9-mers"
    t = pair_table(panel)
    within = t.filter(pl.col("kind") == "within")
    assert within.height == 2 * 45, "two alleles of ten peptides give 45 pairs each"
    assert within.filter(pl.col("lev") > pl.col("hamming")).height == 0, \
        "edit distance cannot exceed position-wise mismatches at equal length"
    for r in within.head(10).to_dicts():                       # symmetry on a sample
        assert lev(r["pep1"], r["pep2"]) == lev(r["pep2"], r["pep1"])
    print("peptide_distance.demo OK  (%d within-allele pairs over %d peptides, all 9-mers; "
          "lev<=hamming holds for every pair)" % (within.height, panel.height))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
