"""The numbers quoted in the Methods prose that no other script emits.
# 2026-08-09

Everything the manuscript states must trace to code. Four claims in the Methods were derived
interactively and typed in by hand, so they could not be re-checked:

  1. what per-epitope de-duplication removes from each resource (VDJdb, TCRvdb, IMMREP25);
  2. that VDJdb already contains the TCRvdb receptors, which is why publicity is defined against
     VDJdb alone;
  3. that the pooled AIRR repertoire is so large that even OLGA sequences nobody has observed fall
     within one substitution of it, which is why the 1-mismatch degree is read against a control
     rather than against zero;
  4. how many of the pairSEQ mock's (and IMMREP25's) 1000 receptors carry a distinct CDR3beta.

This recomputes all four and writes them as macros. (3) enumerates the neighbours of each QUERY
sequence and tests membership in the pool, never the reverse: the pool has ~18M TCRbeta CDR3s and
one wildcard set over it would not fit in memory, whereas a query has only ~19*len candidates.

Usage: python src/methods_numbers.py   # writes appendix/analysis/methods_macros.tex
"""
from __future__ import annotations
import gzip
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from load_data import valid_cdr3, load_tcrvdb, load_immrep22, VDJDB      # noqa: E402
from cohorts import build_cohorts       # noqa: E402

ADAT = os.path.join(REPO, "appendix", "analysis")
AIRR = os.path.join(REPO, "cache", "airr")
AA = "ACDEFGHIKLMNPQRSTVWY"
N_QUERY = 2000          # OLGA queries for the neighbourhood rate; the rate is a proportion, and
                        # 2000 draws put its standard error near 1%, well below the reported gap


def _dedup_loss(df: pd.DataFrame, cdr3: str) -> float:
    """Percentage of RAW records that per-epitope de-duplication removes, one chain.

    Must be measured before build_cohorts, which de-duplicates on the way in -- measuring it on an
    assembled cohort returns 0 by construction.
    """
    df = df[df[cdr3].map(valid_cdr3)]
    if not len(df):
        return float("nan")
    return 100.0 * (1.0 - len(df.drop_duplicates(["epitope", cdr3])) / len(df))


def _pool_cdr3s(fn: str) -> set:
    """Every valid CDR3 of one pooled-AIRR chain file."""
    out = set()
    with gzip.open(os.path.join(AIRR, fn), "rt") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if len(p) > 3 and valid_cdr3(p[3]):
                out.add(p[3])
    return out


def _within_one(queries, pool: set) -> float:
    """Percentage of queries with a Hamming-<=1 match in pool (exact matches excluded)."""
    hit = 0
    for s in queries:
        if any((s[:i] + a + s[i + 1:]) in pool
               for i in range(len(s)) for a in AA if a != s[i]):
            hit += 1
    return 100.0 * hit / len(queries) if len(queries) else float("nan")


def run():
    coh = build_cohorts(include_olga=True)
    m = {}

    # ---- the raw resources, as they are before build_cohorts de-duplicates them ----
    vraw = pd.read_csv(VDJDB, sep="\t", low_memory=False)
    vraw = vraw[(vraw["species"] == "HomoSapiens") & (vraw["mhc.class"] == "MHCI")].copy()
    vraw["epitope"] = vraw["antigen.epitope"].astype(str)
    traw = load_tcrvdb()

    # (1) what per-epitope de-duplication removes from each resource
    for chain, gene, cdr3 in (("A", "TRA", "cdr3a"), ("B", "TRB", "cdr3b")):
        m["mDupVdjdb%s" % chain] = "%.0f" % _dedup_loss(
            vraw[vraw["gene"] == gene].rename(columns={"cdr3": cdr3}), cdr3)
        m["mDupTcrvdb%s" % chain] = "%.0f" % _dedup_loss(traw, cdr3)
    ipr = coh["immrep25_pos"]["paired"]
    m["mDistinctImm"] = "%.0f" % (100.0 * ipr.cdr3b.nunique() / len(ipr))

    # (2) VDJdb contains the TCRvdb receptors, which is why publicity is defined against VDJdb
    for chain, gene, cdr3 in (("A", "TRA", "cdr3a"), ("B", "TRB", "cdr3b")):
        vd = set(vraw[(vraw["gene"] == gene) & vraw["cdr3"].map(valid_cdr3)]["cdr3"])
        tv = set(traw[cdr3][traw[cdr3].map(valid_cdr3)])
        m["mTvIn%s" % chain] = "%d" % len(tv & vd)
        m["mTvN%s" % chain] = "%d" % len(tv)

    # (3) the pooled AIRR repertoire saturates the 1-mismatch neighbourhood
    rng = np.random.default_rng(0)
    for chain, fn in (("A", "human.tra.aa.tsv.gz"), ("B", "human.trb.aa.tsv.gz")):
        pool = _pool_cdr3s(fn)
        m["mAirrN%s" % chain] = "%d" % len(pool)
        q = coh["olga_random"]["long"]
        q = q[q.chain == chain].cdr3.drop_duplicates().to_numpy()
        q = rng.choice(q, min(N_QUERY, len(q)), replace=False)
        m["mOlgaHit%s" % chain] = "%.1f" % _within_one(q, pool)
        print("  TCR%s: pool %d unique CDR3, %s%% of OLGA queries within one substitution"
              % (chain, len(pool), m["mOlgaHit%s" % chain]))

    # (4) distinct CDR3beta among the 1000 mock / benchmark receptors, and the cohort sizes the
    #     Methods quotes when it introduces each set
    for tag, name in (("Ps", "pairseq_mock"), ("Imm", "immrep25_pos"), ("Ii", "immrep22_true")):
        pr = coh[name]["paired"]
        m["mDistB%s" % tag] = "%d" % pr.cdr3b.nunique()
        m["mNPaired%s" % tag] = "%d" % len(pr)

    # (5) records the largest cohort contributes at the >=30-per-epitope floor, quoted by the
    #     homology-graph legend to say why only the VDJdb sets are subsampled there
    for chain in ("A", "B"):
        lc = coh["vdjdb_hq"]["long"]
        lc = lc[lc.chain == chain]
        vc = lc.epitope.value_counts()
        # comma-grouped: the supplement legend that quotes it has no siunitx
        m["mHqQual%s" % chain] = "{:,}".format(int(lc.epitope.isin(vc[vc >= 30].index).sum()))

    with open(os.path.join(ADAT, "methods_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote appendix/analysis/methods_macros.tex")
    for k, v in m.items():
        print("  %-16s %s" % (k, v))


if __name__ == "__main__":
    run()
