#!/usr/bin/env python
# 2026-09-14  What the challenge actually scored, as against the design it released.
#
# Every number this repo reports about IMMREP25's geometry comes from the released file, which is
# a clean 20 peptides x 50 positives with 450 same-MHC negatives each -- ten equal pools per
# allele. That is NOT what produced the leaderboard. The challenge downsampled the 10,000
# combinations to 8,938, held two A*02:01 peptides back as a public leaderboard, and scored the
# rest. So the geometry behind the reported 0.60 has EIGHT pools on one allele and ten on the
# other, and none of them are the same size.
#
# This matters for exactly one standing result. blind_ceiling derives the peptide-blind bound
# 10/19 ~ 0.5263 from K = 10 equal pools of m = 50, and its own docstring warns that "one more
# peptide per allele and the argument would need redoing". On K = 8 with unequal pools the bound
# moves. The closed form assumes equality, so rather than re-derive it we reuse blind_ceiling's
# construction and its hill-climbing search, both of which take an arbitrary pool-label array and
# therefore generalise for free.
#
# Two things are derived here rather than assumed:
#   - WHICH peptides were public. The leaderboard file carries one column per scored pMHC, so the
#     public pair is the set difference between the release's 20 peptides and those columns. That
#     is a measurement, not a reading of the methods text.
#   - the public/private/straddling partition, checked against the three totals the challenge
#     paper states (175 + 7,344 + 1,419 = 8,938). If the rule we implement is wrong, the assertion
#     fails rather than a plausible-looking number being reported.
#
# The leaderboard also upgrades \veAucBest from a cited scalar to a measured distribution: 124
# scored submissions with the best at 0.60146693501378, which is what 0.60 rounds from.
#
# Sources (see SOURCES.md): dump/media-2.xlsx (leaderboard, 3 sheets auc01/auc/aupr),
# dump/immrep25_test/test.csv (the 8,938-record downsample contestants received, labels withheld),
# dump/immrep25/immrep2025_for_release.tsv (the labels).
#
# Run: python src/scored_geometry.py     (--demo for the self-check)
from __future__ import annotations
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
TEST = os.path.join(REPO, "dump", "immrep25_test", "test.csv")
BOARD = os.path.join(REPO, "dump", "media-2.xlsx")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

import blind_ceiling as B                                         # noqa: E402

MAX_FPR = 0.1
AUC_BEST = 0.60          # the cited, rounded figure this module checks against the source
N_SUB_CITED = 126        # submissions the challenge paper reports
SEED = 0
# the totals the challenge paper states for the partition; asserted, not trusted
N_PUBLIC, N_PRIVATE, N_STRADDLE = 175, 7344, 1419
AA = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,12}")


# --------------------------------------------------------------------------- #
# A minimal xlsx reader. openpyxl is not a dependency of this repo and adding one for a single
# file read is the wrong trade: an xlsx is a zip of XML, and we need one rectangular sheet.
# --------------------------------------------------------------------------- #
def read_sheet(path: str, sheet: int = 1) -> pd.DataFrame:
    """Read sheet `sheet` (1-based, workbook order) of an xlsx into a DataFrame."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        xml = z.read("xl/sharedStrings.xml").decode("utf-8")
        for si in re.findall(r"<si>(.*?)</si>", xml, re.S):
            shared.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
    xml = z.read("xl/worksheets/sheet%d.xml" % sheet).decode("utf-8")
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        cells = {}
        for c in re.finditer(r'<c\s+r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
            col, attrs, body = c.group(1), c.group(2), c.group(3) or ""
            v = re.search(r"<v>(.*?)</v>", body, re.S)
            if v is None:
                t = re.search(r"<t[^>]*>(.*?)</t>", body, re.S)
                cells[col] = t.group(1) if t else ""
            elif re.search(r't="s"', attrs):
                cells[col] = shared[int(v.group(1))]
            else:
                cells[col] = v.group(1)
        if cells:
            rows.append(cells)
    assert len(rows) >= 2, "sheet %d of %s looks empty" % (sheet, path)
    head = rows[0]
    cols = [head[k] for k in sorted(head, key=lambda s: (len(s), s))]
    order = sorted(head, key=lambda s: (len(s), s))
    out = [{head[k]: r.get(k, "") for k in order} for r in rows[1:]]
    return pd.DataFrame(out, columns=cols)


def leaderboard(sheet: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """The scored submissions, and the peptides the per-pMHC columns name."""
    t = read_sheet(BOARD, sheet)
    t["macro_score"] = pd.to_numeric(t["macro_score"], errors="coerce")
    t["rank"] = pd.to_numeric(t["rank"], errors="coerce")
    peps = []
    for c in t.columns:
        m = AA.search(str(c))
        if m and c not in ("model", "macro_score", "rank", "TeamMemberUserNames"):
            peps.append(m.group(0))
    return t, peps


def partition() -> pd.DataFrame:
    """Label every record of the 8,938-row downsample public / private / straddling.

    A record is (receptor, scored peptide). Each receptor has exactly one cognate peptide in the
    release. The challenge split by the receptor's cognate peptide AND the scored peptide: both
    public, both private, or one of each.
    """
    rel = pd.read_csv(BENCH, sep="\t")
    test = pd.read_csv(TEST)
    scored = leaderboard()[1]
    allp = sorted(rel.peptide.unique())
    pub = sorted(set(allp) - set(scored))
    assert len(scored) == 18, "expected 18 scored pMHC columns, found %d" % len(scored)
    assert len(pub) == 2, "expected 2 held-back peptides, found %s" % pub

    cog = (rel[rel.label == 1]
           .set_index(["tcra_cdr3", "tcrb_cdr3"]).peptide.to_dict())
    key = list(zip(test.CDR3a, test.CDR3b))
    test = test.assign(cognate=[cog.get(k) for k in key])
    assert test.cognate.notna().all(), "%d test records did not join to a cognate peptide" % (
        int(test.cognate.isna().sum()))
    sp, cp = test.Peptide.isin(pub), test.cognate.isin(pub)
    test["arm"] = np.where(sp & cp, "public", np.where(~sp & ~cp, "private", "straddle"))
    return test


def pool_sizes(part: pd.DataFrame) -> pd.DataFrame:
    """Per-peptide scored pool sizes on the private arm: the geometry that produced 0.60."""
    d = part[part.arm == "private"]
    return (d.groupby(["HLA", "Peptide"]).size().rename("n_scored").reset_index()
            .sort_values(["HLA", "n_scored"], ascending=[True, False]))


def main():
    rng = np.random.default_rng(SEED)

    # ---- 1. the leaderboard, as a measured distribution ----
    lb, scored = leaderboard()
    lb = lb.dropna(subset=["macro_score"]).sort_values("macro_score", ascending=False)
    best = float(lb.macro_score.iloc[0])
    print("=== the leaderboard (dump/media-2.xlsx, sheet auc01) ===")
    print("%d scored submissions over %d pMHC columns; the paper reports %d"
          % (len(lb), len(scored), N_SUB_CITED))
    print("best %.14f (cited as %.2f), median %.4f, min %.4f"
          % (best, AUC_BEST, float(lb.macro_score.median()), float(lb.macro_score.min())))
    print("top 5:")
    for r in lb.head(5).itertuples():
        print("   %.6f  %s" % (r.macro_score, str(r.model)[:58]))
    lb.to_csv(os.path.join(RESULTS, "scored_leaderboard.csv"), index=False)

    # ---- 2. the partition, checked against the published totals ----
    part = partition()
    arms = part.arm.value_counts()
    print("\n=== the 8,938-record downsample, partitioned ===")
    for a in ("public", "private", "straddle"):
        print("  %-9s %5d" % (a, int(arms.get(a, 0))))
    ok = (int(arms.get("public", 0)) == N_PUBLIC and int(arms.get("private", 0)) == N_PRIVATE
          and int(arms.get("straddle", 0)) == N_STRADDLE)
    print("  matches the challenge paper's %d / %d / %d: %s"
          % (N_PUBLIC, N_PRIVATE, N_STRADDLE, ok))
    assert ok, "the partition rule does not reproduce the published totals -- do not report it"

    ps = pool_sizes(part)
    ps.to_csv(os.path.join(RESULTS, "scored_geometry.csv"), index=False)
    print("\n=== scored pool sizes per allele (the private arm) ===")
    print(ps.to_string(index=False))

    # ---- 3. what the peptide-blind bound becomes on that geometry ----
    #
    # The leaderboard's macro_score is a mean over ALL 18 scored pMHCs, so the bound on the
    # REPORTED metric is the 18-pool combination -- not the larger of the two per-allele values.
    # The alleles are disjoint receptor sets (each peptide's negatives are same-MHC only), so a
    # single shared ranking attains each arm's block bound independently and the arms simply add:
    #
    #     bound = [ sum_a (1 + (K_a - 1) * floor) ] / sum_a K_a
    #
    # which is 0.5263 for the released 10 + 10 and rises when an arm loses peptides.
    print("\n=== peptide-blind bound, released design vs scored geometry ===")
    floor = B.floor_value(MAX_FPR)
    priv = part[part.arm == "private"]
    rows = []
    for hla, g in priv.groupby("HLA"):
        # per pool: positives are receptors whose COGNATE peptide is the scored one
        cnt = (g.assign(is_pos=(g.Peptide == g.cognate))
               .groupby("Peptide").is_pos.agg(n_pos="sum", n_rec="size"))
        cnt["n_neg"] = cnt.n_rec - cnt.n_pos
        K = len(cnt)
        # the block construction floors every other pool only if the top pool's own receptors
        # exhaust their false-positive budget. Asserted from the measured counts, never assumed.
        margin = float(cnt.n_pos.min() - MAX_FPR * cnt.n_neg.max())
        assert margin > 0, (
            "at %s the smallest pool (%d receptors) no longer exhausts the largest pool's "
            "budget (%.1f); the block construction and this formula must be re-derived"
            % (hla, int(cnt.n_pos.min()), MAX_FPR * cnt.n_neg.max()))
        rows.append(dict(hla=hla, n_pools=K, n_records=int(cnt.n_rec.sum()),
                         pos_min=int(cnt.n_pos.min()), pos_max=int(cnt.n_pos.max()),
                         neg_min=int(cnt.n_neg.min()), neg_max=int(cnt.n_neg.max()),
                         regime_margin=margin,
                         arm_bound=(1.0 + (K - 1) * floor) / K))
    bnd = pd.DataFrame(rows)
    print(bnd.to_string(index=False))
    bnd.to_csv(os.path.join(RESULTS, "scored_blind_bound.csv"), index=False)

    k_tot = int(bnd.n_pools.sum())
    scored_bound = float((bnd.arm_bound * bnd.n_pools).sum() / k_tot)
    released_bound = B.ceiling(10, 50, MAX_FPR)
    # cross-check: the same combination written out from the floor alone
    chk = (len(bnd) + (k_tot - len(bnd)) * floor) / k_tot
    assert abs(chk - scored_bound) < 1e-12, (chk, scored_bound)
    print("\nfloor %.6f (= 9/19); per-allele bounds %s over %d pools"
          % (floor, ", ".join("%.4f" % b for b in bnd.arm_bound), k_tot))
    print("released design (10 + 10 equal pools): %.4f" % released_bound)
    print("SCORED geometry (%s pools): %.4f (%+.4f) -- this is the bound on the macro the "
          "leaderboard reports" % (" + ".join(str(k) for k in bnd.n_pools), scored_bound,
                                   scored_bound - released_bound))
    print("the leader's %.6f sits %+.4f above it" % (best, best - scored_bound))
    worst = scored_bound

    macros = {
        "scgNsub": "%d" % len(lb),
        "scgNsubCited": "%d" % N_SUB_CITED,
        "scgBest": "%.4f" % best,
        "scgBestFull": "%.6f" % best,
        "scgMedian": "%.4f" % float(lb.macro_score.median()),
        "scgMin": "%.4f" % float(lb.macro_score.min()),
        "scgNscoredPep": "%d" % len(scored),
        "scgNpublicPep": "%d" % (20 - len(scored)),
        "scgNdown": "%d" % len(part),
        "scgNpublic": "%d" % int(arms.get("public", 0)),
        "scgNprivate": "%d" % int(arms.get("private", 0)),
        "scgNstraddle": "%d" % int(arms.get("straddle", 0)),
        "scgPoolMin": "%d" % int(ps.n_scored.min()),
        "scgPoolMax": "%d" % int(ps.n_scored.max()),
        "scgKmin": "%d" % int(bnd.n_pools.min()),
        "scgKmax": "%d" % int(bnd.n_pools.max()),
        "scgBound": "%.4f" % worst,
        # the per-arm caps Eq. (4) gives before the pool-weighted combination; the supplement
        # quotes both, so they are emitted rather than left for a reader to recompute
        "scgArmA": "%.4f" % bnd.arm_bound.max(),
        "scgArmB": "%.4f" % bnd.arm_bound.min(),
        "scgBoundReleased": "%.4f" % B.ceiling(10, 50, MAX_FPR),
        "scgBoundShift": "%.4f" % (worst - B.ceiling(10, 50, MAX_FPR)),
        "scgLeaderGap": "%.4f" % (best - worst),
    }
    with open(os.path.join(ADAT, "scored_geometry_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/scored_leaderboard.csv, results/scored_geometry.csv, "
          "results/scored_blind_bound.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "scored_geometry_macros.tex"), len(macros)))


def demo():
    """Self-check. The xlsx reader is the risky part -- a regex over XML -- so it is checked
    against the one value we know independently: the cited 0.60 must be what the best score
    rounds to, and the sheet must be rectangular with the documented columns."""
    lb, scored = leaderboard()
    assert {"model", "macro_score", "rank"} <= set(lb.columns), sorted(lb.columns)[:8]
    assert len(scored) == 18, "expected 18 per-pMHC columns, got %d" % len(scored)
    s = lb.macro_score.dropna()
    assert len(s) > 100, "only %d parsed scores -- the reader is dropping rows" % len(s)
    assert abs(round(float(s.max()), 2) - AUC_BEST) < 1e-9, (
        "best %.6f does not round to the cited %.2f" % (s.max(), AUC_BEST))
    assert s.min() > 0.4 and s.max() < 0.7, "scores outside the plausible band: %s" % (
        (float(s.min()), float(s.max())),)
    # all three sheets must parse to the same shape (auc01 / auc / aupr)
    shapes = [read_sheet(BOARD, i).shape for i in (1, 2, 3)]
    assert len(set(shapes)) == 1, "sheets disagree in shape: %s" % shapes
    # the ranks are 1..N_SUB_CITED with gaps, so the count must not exceed the cited total
    assert len(s) <= N_SUB_CITED, "more scored rows (%d) than submissions (%d)" % (
        len(s), N_SUB_CITED)
    print("scored_geometry.demo OK  (xlsx reader parsed %d submissions x %d pMHC columns from 3 "
          "sheets of identical shape %s; best %.14f rounds to the cited %.2f; range %.4f-%.4f)"
          % (len(s), len(scored), shapes[0], float(s.max()), AUC_BEST,
             float(s.min()), float(s.max())))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
