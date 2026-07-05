"""Build the beta-only 'expanded clone' control (mlr_prolif) from isalgo/airr_benchmark
alice/mlr proliferating samples -- a real analogue of immrep25's MIRA methodology
(TCRB-only, expanded/proliferating T cells).

The 6 proliferating samples are actually 3 independent MLR reactions, each captured in
two heavily-overlapping duplicate samples (they share ~50-65% of their top clones):
    R1 = {MLR7_TCR1, TCR4}   R2 = {MLR8_TCR2, TCR5}   R3 = {MLR9_TCR3, TCR6}
Each sample has two sequencing replicas (_1, _2), so every reaction has 4 replica files.
We treat each replica file as one virtual "epitope" (its top-200 TCRB clonotypes by read
count) -> 12 epitope points across 3 reactions. In the homology test these points use a
reaction-aware background (a replicate is scored against the *other reactions* only, never
its own siblings, which share the same expanded clones) -- otherwise duplicate/replicate
sharing would manufacture cross-epitope homology. Fetch data with scripts/fetch_mlr.sh.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import valid_cdr3

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLR = os.path.join(REPO, "cache", "airr_mlr")
RESULTS = os.path.join(REPO, "results")

TOP = 200
REACTIONS = {"R1": ["MLR7_TCR1", "TCR4"],
             "R2": ["MLR8_TCR2", "TCR5"],
             "R3": ["MLR9_TCR3", "TCR6"]}


def _top_file(sample, rep, top=TOP):
    """Top-`top` TCRB CDR3s of one replica file (Adaptive/immunoSEQ export).

    Some files carry ragged extra trailing columns, so read positionally:
    col 1 = aminoAcid (CDR3), col 2 = count.
    """
    p = os.path.join(MLR, "%s_Proliferating_%d.tsv.gz" % (sample, rep))
    df = pd.read_csv(p, sep="\t", header=None, skiprows=1, usecols=[1, 2],
                     names=["cdr3", "count"], dtype={1: str}, low_memory=False)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df[df.cdr3.map(valid_cdr3) & df["count"].notna()]
    g = df.groupby("cdr3", as_index=False)["count"].sum()
    return g.sort_values("count", ascending=False).head(top).cdr3.tolist()


def main():
    rows = []
    for rk, samples in REACTIONS.items():
        for s in samples:
            for rep in (1, 2):
                epi = "%s_%d" % (s, rep)
                for i, cdr3 in enumerate(_top_file(s, rep)):
                    rows.append({"cohort": "mlr_prolif", "tcr_id": "%s_%d" % (epi, i),
                                 "chain": "B", "epitope": epi, "cdr3": cdr3, "v": "", "j": "",
                                 "quality": "noise", "reaction": rk})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "mlr_prolif.tsv"), sep="\t", index=False)
    print("built mlr_prolif: %d TCRB clonotypes, %d epitopes (replica files), %d reactions"
          % (len(out), out.epitope.nunique(), out.reaction.nunique()))
    print("epitopes per reaction:")
    print(out.groupby("reaction").epitope.nunique().to_string())


if __name__ == "__main__":
    main()
