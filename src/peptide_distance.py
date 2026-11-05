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


# MHC class I anchors for the two alleles in the panel. Both bind nine-mers through P2 and the
# C-terminal residue, which face the groove rather than the TCR: A*02:01 takes L/M at P2 and
# V/L/I at P9, B*40:01 a near-obligate E at P2 and L/A/W at P9. Those positions are therefore
# shared BY CONSTRUCTION among peptides selected for the same allele, and counting them inflates
# any similarity measure. Reviewer 3 asked whether high-edit-distance peptides might still share
# critical TCR-facing contacts; masking the anchors is what makes that question answerable.
ANCHORS = (1, 8)          # zero-based P2 and P9 of a nine-mer


def tcr_facing(seq: str, anchors=ANCHORS) -> str:
    """Drop the anchor positions, leaving the TCR-facing residues in order.

    Used for the mismatch count, where positions are compared independently and dropping two of
    them simply shortens the comparison to seven.
    """
    return "".join(c for i, c in enumerate(seq) if i not in anchors)


def mask_anchors(seq: str, anchors=ANCHORS, sentinel: str = ".") -> str:
    """Replace the anchor residues with `sentinel`, keeping the sequence's length and spacing.

    Used for the shared-substring measure, where position matters. The two sequences of a pair
    must be masked with DIFFERENT sentinels, so that a masked position never matches and a
    shared run cannot bridge an anchor -- see `lcsubstr_tcr`. Masking both with the same
    character would let two peptides sharing nothing but their groove-facing anchors read as
    carrying a long common motif, which is the artefact this whole arm exists to avoid.
    """
    return "".join(sentinel if i in anchors else c for i, c in enumerate(seq))


def hamming_tcr(a: str, b: str) -> int:
    """Position-wise mismatches among the TCR-facing residues only (at most 7 of 9)."""
    return hamming(tcr_facing(a), tcr_facing(b))


def lcsubstr_tcr(a: str, b: str) -> int:
    """Longest shared contiguous run of TCR-facing residues, never bridging an anchor."""
    return lcsubstr(mask_anchors(a, sentinel="1"), mask_anchors(b, sentinel="2"))


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
                    lambda s: lcsubstr(s["pep1"], s["pep2"]), return_dtype=pl.Int64),
                # the same measures restricted to the TCR-facing positions
                hamming_tcr=pl.struct("pep1", "pep2").map_elements(
                    lambda s: hamming_tcr(s["pep1"], s["pep2"]), return_dtype=pl.Int64),
                lcsubstr_tcr=pl.struct("pep1", "pep2").map_elements(
                    lambda s: lcsubstr_tcr(s["pep1"], s["pep2"]), return_dtype=pl.Int64)))


def summarise(t: pl.DataFrame) -> pl.DataFrame:
    """Tidy summary: one row per (kind, allele, metric)."""
    return (t.unpivot(index=["kind", "hla"], on=["lev", "hamming", "lcsubstr", "hamming_tcr", "lcsubstr_tcr"],
                      variable_name="metric", value_name="value")
            .group_by("kind", "hla", "metric")
            .agg(n_pairs=pl.len(), min=pl.col("value").min(),
                 median=pl.col("value").median(), max=pl.col("value").max())
            .sort("kind", "hla", "metric"))


def main():
    panel = load_panel()
    t = pair_table(panel)
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

    # The TCR-facing arm: identical peptides at the anchors are still unrelated where it
    # matters. n_shared_tcr counts pairs sharing a run of 4+ TCR-facing residues, the length a
    # contiguous recognition motif would need (the GIGI/BST2 case is the only well-characterised
    # precedent, and it rests on such a stretch).
    lcs_tcr_max = int(within["lcsubstr_tcr"].max())
    n_tcr_ge4 = int(within.filter(pl.col("lcsubstr_tcr") >= 4).height)
    ham_tcr_min = int(within["hamming_tcr"].min())

    # Supplementary alignment: the panel with its anchors masked, so a reader can see that the
    # shared residues are the groove-facing ones.
    from paths import ms_path
    panel_sorted = panel.sort("hla", "peptide")
    with open(ms_path("tables", "peptide_alignment_table.tex"), "w") as fh:
        fh.write("%% GENERATED by src/peptide_distance.py -- do not edit.\n"
                 "%% Regenerate: python src/peptide_distance.py "
                 "(from the 2026-immrep25-audit repo).\n")
        fh.write("\\begin{tabular}{ll*{9}{c}}\n\\toprule\n")
        fh.write("HLA & peptide & " + " & ".join("P%d" % (i + 1) for i in range(9))
                 + " \\\\\n\\midrule\n")
        last = None
        for r in panel_sorted.to_dicts():
            hla = r["hla"] if r["hla"] != last else ""
            last = r["hla"]
            cells = []
            for i, c in enumerate(r["peptide"]):
                cells.append("\\textbf{%s}" % c if i in ANCHORS else c)
            fh.write("%s & \\texttt{%s} & %s \\\\\n"
                     % (hla, r["peptide"], " & ".join(cells)))
        fh.write("\\bottomrule\n\\end{tabular}\n")

    print("\nwith the MHC anchors (P2, P9) masked, over the same within-allele pairs:")
    print("  longest shared TCR-facing substring: %d residues" % lcs_tcr_max)
    print("  pairs sharing 4+ contiguous TCR-facing residues: %d of %d" % (n_tcr_ge4, n_pairs))
    print("  fewest TCR-facing mismatches in any pair: %d of 7 positions" % ham_tcr_min)

    macros = {
        "pepNpairs": "%d" % n_pairs,
        "pepAnchorP": "P2 and P9",
        "pepNtcrPos": "%d" % (9 - len(ANCHORS)),
        "pepLcsTcrMax": "%d" % lcs_tcr_max,
        "pepLcsTcrMed": "%.0f" % within["lcsubstr_tcr"].median(),
        "pepNtcrGeFour": "%d" % n_tcr_ge4,
        "pepHamTcrMin": "%d" % ham_tcr_min,
        "pepHamTcrMed": "%.0f" % within["hamming_tcr"].median(),
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
    # masking must blank exactly the two anchors, and the two measures must behave as named
    assert mask_anchors("ABCDEFGHI") == "A.CDEFGH."
    assert tcr_facing("ABCDEFGHI") == "ACDEFGH"
    # peptides differing ONLY at their anchors are identical where the TCR looks
    assert hamming_tcr("ALCDEFGHV", "AMCDEFGHL") == 0
    # ... and a shared run must stop at an anchor rather than run through it. Here P9 is V in
    # both, so the unmasked run "CDEFGHV" bridges it; the TCR-facing run must not.
    assert lcsubstr("AACDEFGHV", "ABCDEFGHV") == 7
    assert lcsubstr_tcr("AACDEFGHV", "ABCDEFGHV") == 6, \
        "a shared run must stop at an anchor, not run through it"
    # two peptides sharing nothing but their anchors must share no TCR-facing run at all
    assert lcsubstr_tcr("ALAAAAAAV", "AMBBBBBBV") == 1        # only P1
    assert hamming_tcr("ALAAAAAAV", "AMBBBBBBV") == 6         # six of seven differ
    masked = within.filter(pl.col("lcsubstr_tcr") > pl.col("lcsubstr")).height
    assert masked == 0, "restricting to TCR-facing positions cannot lengthen a shared run"
    assert within["hamming_tcr"].max() <= 7, "at most seven TCR-facing positions can differ"

    print("peptide_distance.demo OK  (%d within-allele pairs over %d peptides, all 9-mers; "
          "lev<=hamming holds for every pair; anchor masking blanks P2/P9 only and never "
          "lengthens a shared run)" % (within.height, panel.height))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
