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
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(REPO, "dump")

IMMREP = os.path.join(DUMP, "immrep25", "immrep2025_for_release.tsv")
IMMREP22_DIR = os.path.join(DUMP, "immrep22")
IMMREP23_TRAIN = os.path.join(DUMP, "immrep23", "VDJdb_paired_chain.csv")
TCRVDB = os.path.join(DUMP, "matchmakers", "01_05_2025_TCRvdb.csv")
VDJDB = os.path.join(DUMP, "vdjdb-2026-06-03", "vdjdb.txt")

# VDJdb sources excluded from the VDJdb cohorts (see load_vdjdb). Both are removed AFTER the
# HQ/LQ tier is fixed, so the tier definition -- distinct studies per epitope -- is untouched.
VDJDB_EXCLUDE_PHAGE = "PMID:40498839"      # phage-display library of one degenerate TCR
VDJDB_EXCLUDE_10X = "10xgenomics"          # substring match: the 10x dCODE application note

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
    """Explode a paired frame into per-chain long rows, one per distinct clonotype.

    The per-chain reduction is the one that matters for the homology probe: a paired record
    can be unique while its beta chain repeats across several partners, and those repeats
    would otherwise be counted as separate observations of the same receptor. VDJdb's long
    table is reduced the same way in `load_vdjdb`, so every cohort is treated alike.
    """
    a = paired.rename(columns={"cdr3a": "cdr3", "va": "v", "ja": "j"}).assign(chain="A")
    b = paired.rename(columns={"cdr3b": "cdr3", "vb": "v", "jb": "j"}).assign(chain="B")
    out = pd.concat([a[LONG_COLS], b[LONG_COLS]], ignore_index=True)
    out = out[out["cdr3"].map(valid_cdr3)]
    return out.drop_duplicates(["chain", "epitope", "cdr3"]).reset_index(drop=True)


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
# immrep22 — IMMREP2022 true positives (WG6, curated from VDJdb). One tab-sep
# file per epitope (epitope = filename stem); Label==1 rows are true positives.
# --------------------------------------------------------------------------- #
def load_immrep22() -> pd.DataFrame:
    import glob
    frames = []
    for path in sorted(glob.glob(os.path.join(IMMREP22_DIR, "*.txt"))):
        epi = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, sep="\t")
        if "Label" in df.columns:
            df = df[pd.to_numeric(df["Label"], errors="coerce") == 1]
        frames.append(pd.DataFrame({
            "cohort": "immrep22",
            "epitope": epi,
            "cdr3a": df["TRA_CDR3"].astype(str), "va": df["TRAV"].map(canon_gene), "ja": df["TRAJ"].map(canon_gene),
            "cdr3b": df["TRB_CDR3"].astype(str), "vb": df["TRBV"].map(canon_gene), "jb": df["TRBJ"].map(canon_gene),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["tcr_id"] = ["immrep22_%d" % i for i in range(len(out))]
    return out


def load_immrep23() -> pd.DataFrame:
    """The seen-peptide IMMREP benchmark positives: IMMREP23's training set plus IMMREP22's.

    IMMREP23 (Nielsen 2024) distributes its training set as positives only, paired chain, curated
    from VDJdb -- 11312 records over 808 peptides and 43 class-I HLAs. IMMREP22's true positives are
    almost entirely a subset of it (615 of 618 clonotypes, and every distinct CDR3beta), the two
    being curations of VDJdb at different times, so the union is IMMREP23 plus a handful of
    receptors and is named for the later challenge.

    The CDR3a/CDR3b columns of the IMMREP23 release carry IMGT CDR3s with the Cys104/Phe118 anchors
    stripped; CDR3a_extended/CDR3b_extended carry the junction form used everywhere else in this
    repo, and those are the columns read here. Joining on the bare columns instead returns zero
    overlap with IMMREP22 or VDJdb -- a silent failure, not an empty result.
    """
    tr = pd.read_csv(IMMREP23_TRAIN)
    i23 = pd.DataFrame({
        "cohort": "immrep23", "epitope": tr["Peptide"].astype(str),
        "cdr3a": tr["CDR3a_extended"].astype(str), "va": tr["Va"].map(canon_gene),
        "ja": tr["Ja"].map(canon_gene),
        "cdr3b": tr["CDR3b_extended"].astype(str), "vb": tr["Vb"].map(canon_gene),
        "jb": tr["Jb"].map(canon_gene),
    })
    out = pd.concat([load_immrep22().assign(cohort="immrep23"), i23], ignore_index=True)
    out = out[out.cdr3a.map(valid_cdr3) & out.cdr3b.map(valid_cdr3)]
    out = out.drop_duplicates(["epitope", "cdr3a", "cdr3b"]).reset_index(drop=True)
    out["tcr_id"] = ["immrep23_%d" % i for i in range(len(out))]
    return out


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
#   quality: the EPITOPE is reported by >= 2 distinct studies -> HQ, else LQ
#   both cohorts are reduced to distinct clonotypes (see load_vdjdb docstring)
# --------------------------------------------------------------------------- #
def load_vdjdb(hq_min_refs: int = 2):
    """Return (long_all, paired), each reduced to distinct clonotypes per epitope.

    Quality is a property of the *epitope*, not of the individual receptor: an epitope is HQ
    when at least `hq_min_refs` distinct studies report TCRs against it, and LQ when a single
    study does. This asks whether an epitope's repertoire has been characterised independently
    more than once, which is what makes it a trustworthy reference point, and it is decided
    entirely without reference to CDR3 sequence. That last property matters here: a criterion
    applied to individual receptors -- for instance requiring a receptor or its near-neighbours
    to be reported twice -- would select receptors for having near-neighbours and so partly
    manufacture the convergence the homology probe then measures.

    Both cohorts are reduced to distinct (epitope, CDR3) clonotypes. VDJdb aggregates studies,
    so a widely reported receptor otherwise appears many times over and its repeat entries
    would dominate the pair exposures of any similarity statistic; after de-duplication each
    clonotype contributes once, and the split reflects how well an epitope is studied rather
    than how popular a receptor is.
    """
    v = pd.read_csv(VDJDB, sep="\t", low_memory=False)
    v = v[(v["species"] == "HomoSapiens") & (v["mhc.class"] == "MHCI")].copy()
    v["v"] = v["v.segm"].map(canon_gene)
    v["j"] = v["j.segm"].map(canon_gene)
    v["epitope"] = v["antigen.epitope"].astype(str)
    v["chain"] = v["gene"].map({"TRA": "A", "TRB": "B"})

    # epitope-level support: how many distinct studies report this epitope at all. Counted on the
    # FULL table, before the exclusions below, so that removing a source cannot re-tier an epitope.
    epi_refs = v.groupby("epitope")["reference.id"].nunique()
    v["quality"] = np.where(v.epitope.map(epi_refs) >= hq_min_refs, "hq", "lq")

    # Two sources are then removed from the VDJdb cohorts themselves:
    #  * a phage-display library (VDJDB_EXCLUDE_PHAGE) contributing ~29.7k TCRb variants against a
    #    single alpha chain -- a mutagenesis series around one degenerate receptor, not a repertoire,
    #    and by itself 34% of the HQ TCRb records;
    #  * the 10x dCODE dextramer application note (VDJDB_EXCLUDE_10X), which this study analyses as
    #    an INDEPENDENT positive control (Supplementary Note 3). 86.5% of its VDJdb TCRb clonotypes
    #    are the same records, so leaving it in would make that control a subset of the cohorts it
    #    is meant to validate.
    ref = v["reference.id"].astype(str)
    drop = (ref == VDJDB_EXCLUDE_PHAGE) | ref.str.contains(VDJDB_EXCLUDE_10X, regex=False)
    v = v[~drop].copy()

    long_all = v.assign(cohort="vdjdb", tcr_id="vdjdb_" + v.index.astype(str))
    long_all = long_all[["cohort", "tcr_id", "chain", "epitope", "cdr3", "v", "j", "quality"]]
    long_all = long_all[long_all["cdr3"].map(valid_cdr3)].reset_index(drop=True)
    # one row per distinct clonotype (quality is epitope-level, so it is constant per group)
    long_all = long_all.drop_duplicates(["chain", "epitope", "cdr3"]).reset_index(drop=True)

    # paired: pivot complexes (each complex.id has exactly one TRA + one TRB)
    vp = v[v["complex.id"] != 0]
    a = vp[vp["gene"] == "TRA"].set_index("complex.id")
    b = vp[vp["gene"] == "TRB"].set_index("complex.id")
    common = a.index.intersection(b.index)
    paired = pd.DataFrame({
        "cohort": "vdjdb",
        "tcr_id": ["vdjdbc_%s" % c for c in common],
        "epitope": a.loc[common, "epitope"].values,
        "cdr3a": a.loc[common, "cdr3"].values, "va": a.loc[common, "v"].values, "ja": a.loc[common, "j"].values,
        "cdr3b": b.loc[common, "cdr3"].values, "vb": b.loc[common, "v"].values, "jb": b.loc[common, "j"].values,
        "quality": a.loc[common, "quality"].values,   # epitope-level, so both chains agree
    })
    paired = paired[paired["cdr3a"].map(valid_cdr3) & paired["cdr3b"].map(valid_cdr3)]
    paired = paired.drop_duplicates(["epitope", "cdr3a", "cdr3b"]).reset_index(drop=True)
    return long_all, paired


if __name__ == "__main__":
    im = load_immrep()
    print("immrep:", im.shape, "| positives:", int((im.label == 1).sum()))
    tv = load_tcrvdb()
    print("tcrvdb:", tv.shape, "| true:", int((tv.padj < 1e-5).sum()), "| false:", int((tv.padj >= 1e-5).sum()))
    la, pa = load_vdjdb()
    print("vdjdb long:", la.shape, "| by quality:", dict(la.quality.value_counts()))
    print("vdjdb paired:", pa.shape, "| by quality:", dict(pa.quality.value_counts()))
