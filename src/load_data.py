"""Loaders that normalize each raw dataset into two common schemas.

long (per-chain, for the homology test):
    cohort, tcr_id, chain ('A'|'B'), epitope, cdr3, v, j, quality
paired (wide, for the pairing test):
    cohort, tcr_id, epitope, cdr3a, va, ja, cdr3b, vb, jb, quality

Gene tokens are only allele-stripped (e.g. TRBV19*01 -> TRBV19); no cross-dataset
harmonization is done because every S/N metric is computed *within* a cohort
against its own background, so internal consistency is all that is required.
"""
from __future__ import annotations
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(REPO, "dump")

IMMREP = os.path.join(DUMP, "immrep25", "immrep2025_for_release.tsv")
TCRVDB = os.path.join(DUMP, "matchmakers", "01_05_2025_TCRvdb.csv")
VDJDB = os.path.join(DUMP, "vdjdb-2026-06-03", "vdjdb.txt")

LONG_COLS = ["cohort", "tcr_id", "chain", "epitope", "cdr3", "v", "j", "quality"]
PAIR_COLS = ["cohort", "tcr_id", "epitope", "cdr3a", "va", "ja",
             "cdr3b", "vb", "jb", "quality"]

# valid amino-acid CDR3 (drop sequences with stops/gaps/lowercase)
_AA = set("ACDEFGHIKLMNPQRSTVWY")


def canon_gene(x) -> str:
    """Strip allele and whitespace; keep the native family/gene token."""
    if pd.isna(x):
        return ""
    return str(x).split("*")[0].strip()


def valid_cdr3(s) -> bool:
    if not isinstance(s, str) or len(s) < 5:
        return False
    return set(s).issubset(_AA)


def _pair_to_long(paired: pd.DataFrame) -> pd.DataFrame:
    """Explode a paired frame into per-chain long rows."""
    a = paired.rename(columns={"cdr3a": "cdr3", "va": "v", "ja": "j"}).assign(chain="A")
    b = paired.rename(columns={"cdr3b": "cdr3", "vb": "v", "jb": "j"}).assign(chain="B")
    out = pd.concat([a[LONG_COLS], b[LONG_COLS]], ignore_index=True)
    return out[out["cdr3"].map(valid_cdr3)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# immrep25
# --------------------------------------------------------------------------- #
def load_immrep() -> pd.DataFrame:
    df = pd.read_csv(IMMREP, sep="\t")
    df = df.assign(
        cohort="immrep25",
        tcr_id=["immrep_%d" % i for i in range(len(df))],
        epitope=df["peptide"].astype(str),
        cdr3a=df["tcra_cdr3"].astype(str), va=df["tcra_v"].map(canon_gene), ja=df["tcra_j"].map(canon_gene),
        cdr3b=df["tcrb_cdr3"].astype(str), vb=df["tcrb_v"].map(canon_gene), jb=df["tcrb_j"].map(canon_gene),
        label=df["label"].astype(int),
    )
    return df


# --------------------------------------------------------------------------- #
# TCRvdb (matchmakers) — must be parsed with pandas (embedded commas)
# --------------------------------------------------------------------------- #
def load_tcrvdb() -> pd.DataFrame:
    df = pd.read_csv(TCRVDB)
    df.columns = [c.strip() for c in df.columns]
    padj = pd.to_numeric(df["padj"], errors="coerce")
    out = pd.DataFrame({
        "cohort": "tcrvdb",
        "tcr_id": ["tcrvdb_%d" % i for i in range(len(df))],
        "epitope": df["epitope_aa"].astype(str),
        "cdr3a": df["cdr3_alpha_aa"].astype(str), "va": df["TRAV"].map(canon_gene), "ja": df["TRAJ"].map(canon_gene),
        "cdr3b": df["cdr3_beta_aa"].astype(str), "vb": df["TRBV"].map(canon_gene), "jb": df["TRBJ"].map(canon_gene),
        "padj": padj,
    })
    return out


# --------------------------------------------------------------------------- #
# VDJdb — human, MHC class I. Returns (long_all, paired) with quality flags.
#   single-chain quality: distinct reference.id per (TCR_hash, epitope) >= 2 -> HQ
#   paired quality:       distinct reference.id per (paired clonotype, epitope) >= 2 -> HQ
# --------------------------------------------------------------------------- #
def load_vdjdb(hq_min_refs: int = 2):
    """Return (long_all, paired). HQ is defined per chain (TRA/TRB separately) as a
    (TCR_hash, epitope) supported by >= hq_min_refs distinct references; a paired
    complex is HQ if *either* chain is HQ, and both of its chains inherit that flag
    in the single-chain `long` table.
    """
    v = pd.read_csv(VDJDB, sep="\t", low_memory=False)
    v = v[(v["species"] == "HomoSapiens") & (v["mhc.class"] == "MHCI")].copy()
    v["v"] = v["v.segm"].map(canon_gene)
    v["j"] = v["j.segm"].map(canon_gene)
    v["epitope"] = v["antigen.epitope"].astype(str)
    v["chain"] = v["gene"].map({"TRA": "A", "TRB": "B"})

    # per-chain HQ flag
    nref = v.groupby(["TCR_hash", "epitope"])["reference.id"].transform("nunique")
    v["chain_hq"] = nref >= hq_min_refs

    # complex HQ = any chain HQ (map back onto both rows of each complex)
    cx = v[v["complex.id"] != 0]
    complex_hq = cx.groupby("complex.id")["chain_hq"].transform("any")
    v.loc[cx.index, "complex_hq"] = complex_hq
    # long quality: paired rows use complex HQ (any chain); unpaired use own chain HQ
    is_paired = v["complex.id"] != 0
    hq_flag = v["chain_hq"].where(~is_paired, v["complex_hq"].fillna(False).astype(bool))
    v["quality"] = ["hq" if q else "lq" for q in hq_flag.fillna(False)]

    long_all = v.assign(cohort="vdjdb", tcr_id="vdjdb_" + v.index.astype(str))
    long_all = long_all[["cohort", "tcr_id", "chain", "epitope", "cdr3", "v", "j", "quality"]]
    long_all = long_all[long_all["cdr3"].map(valid_cdr3)].reset_index(drop=True)

    # paired: pivot complexes (each complex.id has exactly one TRA + one TRB)
    vp = v[v["complex.id"] != 0]
    a = vp[vp["gene"] == "TRA"].set_index("complex.id")
    b = vp[vp["gene"] == "TRB"].set_index("complex.id")
    common = a.index.intersection(b.index)
    pair_hq = (a.loc[common, "chain_hq"].values | b.loc[common, "chain_hq"].values)
    paired = pd.DataFrame({
        "cohort": "vdjdb",
        "tcr_id": ["vdjdbc_%s" % c for c in common],
        "epitope": a.loc[common, "epitope"].values,
        "cdr3a": a.loc[common, "cdr3"].values, "va": a.loc[common, "v"].values, "ja": a.loc[common, "j"].values,
        "cdr3b": b.loc[common, "cdr3"].values, "vb": b.loc[common, "v"].values, "jb": b.loc[common, "j"].values,
        "quality": ["hq" if q else "lq" for q in pair_hq],
    })
    paired = paired[paired["cdr3a"].map(valid_cdr3) & paired["cdr3b"].map(valid_cdr3)].reset_index(drop=True)
    return long_all, paired


if __name__ == "__main__":
    im = load_immrep()
    print("immrep:", im.shape, "| positives:", int((im.label == 1).sum()))
    tv = load_tcrvdb()
    print("tcrvdb:", tv.shape, "| true:", int((tv.padj < 1e-5).sum()), "| false:", int((tv.padj >= 1e-5).sum()))
    la, pa = load_vdjdb()
    print("vdjdb long:", la.shape, "| by quality:", dict(la.quality.value_counts()))
    print("vdjdb paired:", pa.shape, "| by quality:", dict(pa.quality.value_counts()))
