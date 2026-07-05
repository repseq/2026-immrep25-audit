"""Build two AIRR real-repertoire controls:

  airr_control  (AIRR random)     -- uniform over unique clonotypes of the pooled human
                                     control (isalgo/airr_control), RANDOM epitope labels.
  airr_top      (AIRR non-random) -- the 20 largest SRA samples (>=50 TRA & >=50 TRB
                                     clonotypes) from isalgo/airr_benchmark; each donor
                                     contributes its top 50 TRA and top 50 TRB clonotypes
                                     (by read count) as one virtual epitope.

The chosen donors are recorded in results/airr_donors.tsv. Fetch data with
scripts/fetch_airr.sh first.
"""
import os, sys, gzip, glob, random
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import valid_cdr3, canon_gene
from cohorts import build_cohorts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIRR = os.path.join(REPO, "cache", "airr")
SRA = os.path.join(REPO, "cache", "airr_bench", "samples")
META = os.path.join(REPO, "cache", "airr_bench", "meta.tsv")
RESULTS = os.path.join(REPO, "results")


# --------------------------------------------------------------------------- #
# AIRR random (pooled control, uniform over unique clonotypes)
# --------------------------------------------------------------------------- #
def reservoir(path, n, seed):
    rng = random.Random(seed)
    res, k = [], 0
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7 or not valid_cdr3(p[3]):
                continue
            rec = (p[3], canon_gene(p[4]), canon_gene(p[6]))
            if len(res) < n:
                res.append(rec)
            else:
                r = rng.randint(0, k)
                if r < n:
                    res[r] = rec
            k += 1
    return pd.DataFrame(res, columns=["cdr3", "v", "j"])


# --------------------------------------------------------------------------- #
# AIRR non-random (20 largest SRA donor samples, top-50 clonotypes each)
# --------------------------------------------------------------------------- #
def _sra_chain(v, j):
    for x in (str(v), str(j)):
        if x.startswith("TRA"):
            return "A"
        if x.startswith("TRB"):
            return "B"
    return None


def load_sra(path):
    df = pd.read_csv(path, sep="\t", usecols=["duplicate_count", "junction_aa", "v_call", "j_call"])
    df["ch"] = [_sra_chain(v, j) for v, j in zip(df.v_call, df.j_call)]
    df = df[df.junction_aa.map(valid_cdr3)]
    return df


def select_donors(n=20, min_each=50):
    rows = []
    for f in glob.glob(os.path.join(SRA, "*.tsv")):
        try:
            df = load_sra(f)
            na = int((df.ch == "A").sum()); nb = int((df.ch == "B").sum())
        except Exception:
            continue
        if na >= min_each and nb >= min_each:
            rows.append((os.path.basename(f)[:-4], na, nb, na + nb))
    r = pd.DataFrame(rows, columns=["Run", "nTRA", "nTRB", "nTR"]).sort_values(
        ["nTR", "Run"], ascending=[False, True]).head(n).reset_index(drop=True)
    return r


def top_clonotypes(df, ch, top):
    sub = df[df.ch == ch].sort_values("duplicate_count", ascending=False).head(top)
    return pd.DataFrame({"cdr3": sub.junction_aa.values,
                         "v": [canon_gene(x) for x in sub.v_call],
                         "j": [canon_gene(x) for x in sub.j_call]})


def _assemble(a, b, name, epis, rng):
    n = min(len(a), len(b)); a, b = a.iloc[:n], b.iloc[:n]
    order = rng.permutation(n)
    return pd.DataFrame({
        "cohort": name, "tcr_id": ["%s_%d" % (name, i) for i in range(n)],
        "epitope": epis[:n],
        "cdr3a": a.cdr3.values, "va": a.v.values, "ja": a.j.values,
        "cdr3b": b.cdr3.values[order], "vb": b.v.values[order], "jb": b.j.values[order],
    })


if __name__ == "__main__":
    imm = build_cohorts()["immrep25_pos"]["paired"]
    n_tcr, n_ep = len(imm), imm.epitope.nunique()
    per = n_tcr // n_ep

    # --- AIRR random ---
    fa = os.path.join(AIRR, "human.tra.aa.tsv.gz"); fb = os.path.join(AIRR, "human.trb.aa.tsv.gz")
    ua, ub = reservoir(fa, n_tcr, 42), reservoir(fb, n_tcr, 43)
    rand_ep = ["airr_ep%02d" % (i % n_ep) for i in range(n_tcr)]
    _assemble(ua, ub, "airr_control", rand_ep, np.random.default_rng(42)).to_csv(
        os.path.join(RESULTS, "airr_control.tsv"), sep="\t", index=False)

    # --- AIRR non-random: 20 largest SRA donors, top-50 TRA/TRB each ---
    donors = select_donors(n=n_ep, min_each=per)
    A_parts, B_parts, epis = [], [], []
    for _, d in donors.iterrows():
        df = load_sra(os.path.join(SRA, d.Run + ".tsv"))
        A_parts.append(top_clonotypes(df, "A", per))
        B_parts.append(top_clonotypes(df, "B", per))
        epis += [d.Run] * per
    ta = pd.concat(A_parts, ignore_index=True); tb = pd.concat(B_parts, ignore_index=True)
    _assemble(ta, tb, "airr_top", epis, np.random.default_rng(44)).to_csv(
        os.path.join(RESULTS, "airr_top.tsv"), sep="\t", index=False)

    # record which donors were used (+ PMID / BioProject)
    meta = pd.read_csv(META, sep="\t").drop_duplicates("Run")
    donors.merge(meta, on="Run", how="left").to_csv(
        os.path.join(RESULTS, "airr_donors.tsv"), sep="\t", index=False)
    print("built airr_control (random) + airr_top (%d SRA donors x top-%d TRA/TRB)"
          % (len(donors), per))
    print("donors:", ", ".join(donors.Run))
