#!/usr/bin/env python
# 2026-09-15  Is binder/non-binder learnable from learned embeddings -- on VDJdb, and on IMMREP25,
# under ONE matched geometry?
#
# WHY THIS EXISTS. Every other probe in this study reports an ABSENCE on IMMREP25, and an absence
# invites the answer "a better method would find it". This is the first read-out that is a PRESENCE
# SOMEWHERE ELSE under an identical geometry: the same 161-parameter model, the same allele, the
# same peptide length, the same 1:9 ratio and the same metric buy 690 nats on VDJdb and -0.08 nats
# on IMMREP25. Probe power stops being an argument and becomes a measurement.
#
# WHY A*02:01 ONLY, AND WHY IT IS NOT A COMPROMISE. Measured on load_vdjdb()'s paired frame: only
# HLA-A*02:01 carries ten 9-mer epitopes at >=40 paired complexes (11 in HQ; every other allele has
# <=5). IMMREP25 has exactly ten A*02:01 9-mer peptides over 500 receptors and 5,000 rows at 1:9. So
# the two datasets match simultaneously on allele, peptide length, panel size, positive:negative
# ratio and receptor count. IMMREP25's B*40:01 half is deliberately OUT OF SCOPE -- it has no VDJdb
# counterpart at this depth, so folding it in would break the one axis this probe controls.
#
# FOUR THINGS THIS FILE'S STRUCTURE IS LOAD-BEARING FOR -- do not "simplify" any of them away:
#
# 1. THE ALLELE IS ASSIGNED BY THE MODAL RULE, NOT THE LEXICOGRAPHIC MINIMUM. transfer_germline's
#    panels() takes the minimum normalised allele per epitope. Because "A*02:01" sorts before
#    "B*07:02" and "B*08:01", a single ambiguous A*02 record captures an epitope: FLRGRAYGL (0% of
#    its records A*02), QAKWRLQTL (3%) and RPPIFIRRL (8%) all enter a supposedly A*02:01 panel under
#    that rule. They are B*08:01/B*08:01/B*07:02 epitopes. That destroys the only property the panel
#    exists to have -- that negatives are SAME-MHC re-pairings, as IMMREP25 builds them. We assign
#    the allele carrying most of an epitope's records instead, and assert the purity. Note "A*02" and
#    "A*02:01" are merged first: the same HLA-A2 restriction annotated at two resolutions, which is
#    already how this project settled the TCRvdb allele against this same dump.
#
# 2. THE PEPTIDE SIDE ENTERS ONLY THROUGH PRODUCTS, AND THE GUARD IS IN THE MAIN PATH. Within a
#    per-peptide AUC the peptide is constant, so a marginal peptide column is rank-irrelevant by
#    construction -- measured: receptor_only and marginal GaussianNB agreed to six decimals on both
#    chains. A design of marginal PCs alone therefore collapses to receptor-only, which the rank
#    identity pins at EXACTLY 0.5. embedding_comparison.py's distinct-rows assert lives only in
#    demo(), so a normal run that lost its peptide columns would emit a plausible table of 0.5000s.
#    Ours runs on every fit.
#
# 3. THE PCA BASIS NEVER SEES A BENCHMARK SEQUENCE. It is fitted on OLGA-generated receptors and
#    IEDB A*02:01 9-mers, persisted, then applied -- esm_pca.py's pattern. This is stronger than
#    embedding_comparison.py, which fits PCA transductively on all 1,000 benchmark receptors and has
#    to defend that as label-leakage-free. Here the question does not arise.
#
# 4. A PEPTIDE "CLEARS THE LEADER" ONLY IF ITS 95% CI EXCLUDES THE LEADER'S SCORE. The author's
#    ruling, and it is stricter than comparing point estimates: IMMREP25's best single peptide
#    (0.6026 on 500 rows, 50 positives) sits numerically above \scgBestFull 0.601467 while being
#    entirely consistent with it. Positives and negatives are resampled SEPARATELY so n_pos and
#    n_neg are held at the realised design -- the McClish partial AUC integrates over FPR<=MAX_FPR,
#    so its grid resolution depends on the class counts, and letting them drift would confound CI
#    width with the very ratio the matched geometry exists to hold fixed. SCOPE, stated because it
#    bounds the claim: this is the sampling variability of the EVALUATION set given the fitted
#    model. It does not propagate training variability.
#
# Read-outs, three on one fit, because AUC alone cannot say whether features earn their parameters:
#   macro-AUC0.1  the benchmark's own metric (McClish-standardised, max_fpr=0.1)
#   G             1 - CE/CE_chance, panel2_ce.py's normalised information gain. Both datasets are
#                 held at 1:9 so CE_chance is a shared constant and G is directly comparable --
#                 which is what the matched subsampling buys. G>0 = the representation pays rent.
#   dAIC          2k - 2(logL - logL_null) against the intercept-only model. Negative = the
#                 representation earns its k parameters. k is identical across datasets by design.
# Each with the epitope SEEN in training (grouped 5-fold, no receptor crosses a fold) and HELD OUT
# (strict leave-one-epitope-out: the held-out epitope's cognate receptors are withheld too).
#
# Run: .venv/bin/python src/learnability.py --prep     # writes $LEARNABILITY_WORK/*.csv
#      .venv-embed/bin/python src/learnability_embed.py
#      .venv/bin/python src/learnability.py --score
#      .venv/bin/python src/learnability.py --macros   # re-emit from the committed CSVs, no refit
#      .venv/bin/python src/learnability.py --demo
#
# Writing the supplementary table needs AUDIT_MS_REPO when working in a git worktree, or it lands
# in the main manuscript checkout where the caller cannot see it (see src/paths.py).
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
# Staged embeddings and PCA bases. Env-var-with-default, as panel1_q.py does with $PANEL1_WORK:
# these are large intermediates, not results, and they must not sit in the repo.
WORK = os.environ.get("LEARNABILITY_WORK", os.path.expanduser("~/tmp/learnability"))
CACHE_EMBED = os.path.join(REPO, "cache", "embed")
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
VDJDB_RAW = os.path.join(REPO, "dump", "vdjdb-2026-06-03", "vdjdb.txt")
OLGA_POOL = os.path.join(REPO, "cache", "olga_pool_%s.tsv")
IEDB = os.path.expanduser("~/hf/pmhc_data/pmhc/pmhc_full.tsv.gz")

SEED = 0
ALLELE = "A*02:01"
PANEL = 10          # epitopes per panel; IMMREP25 uses ten per allele
NEG = PANEL - 1     # negatives per positive, as IMMREP25 forms them
MIN_PAIRS = 40      # paired complexes an epitope needs to qualify (transfer_germline's MIN_REC)
CAP = 200           # receptors kept per epitope; VDJdb is skewed (max 10,228 here)
N_BG = 1000         # background receptors and background peptides, per the author's spec
R_PC = 50           # receptor principal components
P_PC = 10           # peptide principal components (rank-inert marginally; see note 2)
KR = KE = 10        # leading PCs entering the product block
MAX_FPR = 0.1
N_BOOT = 1000       # bootstrap resamples for the per-peptide CI (esm_cluster/structure_signal's B)
# The leaderboard best at full precision. \scgBestFull, emitted by scored_geometry.py -- NOT
# re-emitted here: a second macro for one published quantity is how two values of it reach one paper.
AUC_BEST_FULL = 0.601467
MIN_AA = 5          # low-complexity guard for background peptides (distinct residues)
MIN_PURITY = 0.90   # an epitope must carry this share of its records under the panel allele

# SCEPTR requires V symbols that are standardised AND FUNCTIONAL, and raises BadV otherwise -- an
# unmapped gene reaching the model as a null would silently delete that receptor's V channel, the
# failure embed_cache.py's header documents. VDJdb's canon_gene only strips the allele, so it leaves
# non-canonical gene names; IMMREP25 never hits this because it routes through adaptive_imgt_map.tsv,
# which already emits canonical symbols. Both corrections below were derived by running
# tidytcells.tr.standardize(enforce_functional=True, precision='gene') over every distinct token in
# all three receptor sets: IMMREP25's 1,000 and the 1,000 OLGA background receptors need NEITHER,
# which is the control that says this is a VDJdb annotation issue and not a bug here.
# tidytcells lives only in .venv-embed, so the corrections are applied here as an explicit map.
V_RESPELL = {"TRAV14": "TRAV14/DV4", "TRAV23": "TRAV23/DV6", "TRAV29": "TRAV29/DV5"}
V_DEAD = ("TRBV23-1",)      # no functional alleles; the carrying receptor is dropped, not imputed

REP_LABEL = {"sceptr": "SCEPTR", "esm": "ESM-2 (35M)"}
ARM_LABEL = {"seen": "seen in training", "unseen": "held out"}


# --------------------------------------------------------------------------- #
# allele assignment
# --------------------------------------------------------------------------- #
def _norm_allele(s) -> str:
    """`HLA-A*02:01,HLA-A*02` -> `A*02:01`, then A*02 -> A*02:01. Mirrors transfer_germline._allele
    except for the final merge: A*02 and A*02:01 are one restriction at two annotation depths."""
    first = str(s).split(",")[0]
    if first.startswith("HLA-"):
        first = first[4:]
    out = ":".join(first.split(":")[:2])
    return ALLELE if out == "A*02" else out


def epitope_alleles() -> pd.DataFrame:
    """Per epitope: its MODAL allele and that allele's share of its records (see note 1)."""
    raw = pd.read_csv(VDJDB_RAW, sep="\t", low_memory=False,
                      usecols=["species", "mhc.class", "mhc.a", "antigen.epitope"])
    raw = raw[(raw["species"] == "HomoSapiens") & (raw["mhc.class"] == "MHCI")].copy()
    raw["al"] = raw["mhc.a"].map(_norm_allele)
    cnt = raw.groupby(["antigen.epitope", "al"]).size().rename("n").reset_index()
    tot = cnt.groupby("antigen.epitope")["n"].sum().rename("tot")
    top = (cnt.sort_values(["antigen.epitope", "n"], ascending=[True, False])
              .groupby("antigen.epitope").first().join(tot))
    top["purity"] = top.n / top.tot
    return top.rename(columns={"al": "allele"})[["allele", "purity"]]


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def vdjdb_panel(rng) -> tuple[pd.DataFrame, list[str]]:
    """The PANEL deepest HQ A*02:01 9-mer epitopes, capped at CAP paired receptors each."""
    from load_data import load_vdjdb
    _, paired = load_vdjdb()
    al = epitope_alleles()
    p = paired[(paired.epitope.str.len() == 9) & (paired.quality == "hq")].copy()
    p = p.join(al, on="epitope")
    assert p.allele.notna().all() and p.purity.notna().all(), "an epitope has no allele record"
    p = p[p.allele == ALLELE]
    # The modal rule can be won on a bare majority: the pool's minimum purity is 0.500. Ambiguous
    # epitopes are therefore excluded BEFORE ranking by depth, not caught afterwards -- a deep but
    # 60%-A2 epitope would otherwise enter the panel and break the same-MHC negative construction
    # the modal rule exists to protect. The selected panel's purity is then asserted as a guard.
    p = p[p.purity >= MIN_PURITY]

    n = p.groupby("epitope").size().rename("n").reset_index()
    ok = n[n.n >= MIN_PAIRS].sort_values(["n", "epitope"], ascending=[False, True])
    assert len(ok) >= PANEL, "only %d A*02:01 9-mer epitopes at >=%d pairs" % (len(ok), MIN_PAIRS)
    eps = ok.epitope.head(PANEL).tolist()
    assert p[p.epitope.isin(eps)].purity.min() > 0.99, \
        "selected panel is not allele-pure: min purity %.4f" % p[p.epitope.isin(eps)].purity.min()

    # canonicalise V symbols and drop the non-functional ones BEFORE capping, so the cap is applied
    # to receptors that will actually embed and the retained count is the count that gets used
    p = p.copy()
    p["va"] = p.va.replace(V_RESPELL)
    p["vb"] = p.vb.replace(V_RESPELL)
    n_row0 = len(p)
    n_rec0 = (p.cdr3a + "|" + p.cdr3b).nunique()
    p = p[~p.va.isin(V_DEAD) & ~p.vb.isin(V_DEAD)]
    print("  V genes: dropped %d of %d (epitope, receptor) rows and %d of %d distinct receptors "
          "carrying a non-functional gene %s; %d rows retained"
          % (n_row0 - len(p), n_row0, n_rec0 - (p.cdr3a + "|" + p.cdr3b).nunique(), n_rec0,
             list(V_DEAD), len(p)))

    out = []
    for ep in sorted(eps):
        sub = p[p.epitope == ep].sort_values(["cdr3a", "cdr3b"]).reset_index(drop=True)
        if len(sub) > CAP:
            sub = sub.iloc[np.sort(rng.choice(len(sub), CAP, replace=False))]
        out.append(sub)
    rec = pd.concat(out, ignore_index=True)
    rec["key"] = rec.cdr3a + "|" + rec.cdr3b
    # one receptor may appear under two epitopes; keep the pairing but make the key unique per row
    return rec, sorted(eps)


def immrep_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """IMMREP25's A*02:01 half: 500 receptors, 10 peptides, 5,000 rows, already 1:9."""
    d = pd.read_csv(BENCH, sep="\t")
    d = d[d.hla == ALLELE].reset_index(drop=True)
    assert len(d) == 5000 and int(d.label.sum()) == 500, (len(d), int(d.label.sum()))
    key = pd.read_csv(os.path.join(CACHE_EMBED, "receptor_key.csv"))
    join = ["tcra_cdr3", "tcrb_cdr3", "tcra_v", "tcrb_v", "tcra_j", "tcrb_j"]
    key["kidx"] = np.arange(len(key))
    d = d.merge(key[join + ["kidx"]], on=join, how="left")
    assert d.kidx.notna().all(), "%d rows did not match cache/embed/receptor_key.csv" % \
        int(d.kidx.isna().sum())
    d["kidx"] = d.kidx.astype(int)
    d["key"] = d.tcra_cdr3 + "|" + d.tcrb_cdr3
    peps = sorted(d.peptide.unique())
    assert len(peps) == PANEL
    return d, key, peps


def vdjdb_rows(rec: pd.DataFrame, eps: list[str], rng) -> pd.DataFrame:
    """IMMREP25's row construction on VDJdb: each epitope's receptors are its positives and the
    other nine epitopes' receptors its negatives -- forced to EXACTLY NEG negatives per positive.

    POSITIVES ARE CAPPED TOO, and that is the point. Subsampling only the negatives (as
    transfer_germline.match_ratio does) does not reach 1:9 here: a deep epitope has 200 positives
    against a 1,578-receptor negative pool, i.e. 7.9x, so the cap never binds and the realised
    fraction came out 0.1111 against IMMREP25's 0.1000, with ragged per-peptide ratios on the two
    shallow epitopes. CE_chance is a function of the positive fraction, so G is comparable across
    datasets ONLY if the fractions match exactly -- which is what the matched geometry buys and
    what demo() now asserts. Losing positives on the deep epitopes is the correct trade.
    """
    by = {e: g.reset_index(drop=True) for e, g in rec.groupby("epitope")}
    frames = []
    for ep in eps:
        pos_all = by[ep]
        neg_all = pd.concat([by[o] for o in eps if o != ep], ignore_index=True)
        k = min(len(pos_all), len(neg_all) // NEG)          # positives this peptide can support
        assert k > 0, "%s: negative pool too small (%d)" % (ep, len(neg_all))
        pos = pos_all.iloc[np.sort(rng.choice(len(pos_all), k, replace=False))]
        neg = neg_all.iloc[np.sort(rng.choice(len(neg_all), NEG * k, replace=False))]
        frames.append(pos.assign(peptide=ep, label=1))
        frames.append(neg.assign(peptide=ep, label=0))
    out = pd.concat(frames, ignore_index=True)
    return out[["peptide", "epitope", "key", "cdr3a", "va", "cdr3b", "vb", "label"]]


# --------------------------------------------------------------------------- #
# background pools -- the PCA basis, and it must contain no benchmark sequence
# --------------------------------------------------------------------------- #
def olga_background(banned_a: set, banned_b: set, rng) -> pd.DataFrame:
    """N_BG paired OLGA receptors, re-drawn to exclude every benchmark CDR3.

    results/olga_random.tsv is 1,000 paired OLGA TCRs already, but 4 of its alpha CDR3s collide
    with IMMREP25, so it is not basis-clean. The 100k-per-chain pools have 88,640/99,344 distinct
    sequences, so re-pairing from them costs nothing and needs no OLGA run.
    """
    cols = ["cdr3", "v", "j", "pgen"]
    a = pd.read_csv(OLGA_POOL % "A", sep="\t", names=cols)
    b = pd.read_csv(OLGA_POOL % "B", sep="\t", names=cols)
    a = a[~a.cdr3.isin(banned_a)].drop_duplicates("cdr3").reset_index(drop=True)
    b = b[~b.cdr3.isin(banned_b)].drop_duplicates("cdr3").reset_index(drop=True)
    assert len(a) >= N_BG and len(b) >= N_BG, (len(a), len(b))
    ia = np.sort(rng.choice(len(a), N_BG, replace=False))
    ib = np.sort(rng.choice(len(b), N_BG, replace=False))
    out = pd.DataFrame({"CDR3A": a.cdr3.values[ia], "TRAV": a.v.values[ia],
                        "CDR3B": b.cdr3.values[ib], "TRBV": b.v.values[ib]})
    out["key"] = out.CDR3A + "|" + out.CDR3B
    return out


def iedb_background(banned: set, rng) -> pd.DataFrame:
    """N_BG A*02:01-restricted 9-mer IEDB epitopes, none of them a benchmark peptide.

    Source is ~/hf/pmhc_data/pmhc/pmhc_full.tsv.gz, the `full` tier: 1,482,188 positive IEDB
    epitope-MHC records, experimental, nothing predicted (see that directory's SOURCES.md). The
    upstream doc calls the `shortlist` tier (>=2 references) "the better background and the worse
    coverage"; `full` is used here because the pool only has to SPAN a basis, for which coverage is
    what matters and no label is ever read. Allele-matched on purpose: the basis should span the
    peptide space the task actually lives in. MIN_AA drops homopolymers (the file opens with
    AAAAAAAAA).
    """
    t = pd.read_csv(IEDB, sep="\t", usecols=["epitope", "mhc_a", "mhc_class", "mhc_species"])
    t = t[(t.mhc_class == "MHCI") & (t.mhc_species == "HomoSapiens")
          & (t.epitope.str.len() == 9)
          & (t.mhc_a.astype(str).str.startswith("HLA-A*02:01"))]
    pep = pd.Series(sorted(set(t.epitope) - banned))
    pep = pep[pep.map(lambda s: set(s).issubset(set("ACDEFGHIKLMNPQRSTVWY"))
                      and len(set(s)) >= MIN_AA)]
    assert len(pep) >= N_BG, "only %d clean background peptides" % len(pep)
    return pd.DataFrame({"peptide": pep.iloc[np.sort(rng.choice(len(pep), N_BG, replace=False))]
                        .reset_index(drop=True)})


# --------------------------------------------------------------------------- #
# prep
# --------------------------------------------------------------------------- #
def prep():
    os.makedirs(WORK, exist_ok=True)
    rng = np.random.default_rng(SEED)

    imm, key, imm_peps = immrep_panel()
    vrec, veps = vdjdb_panel(np.random.default_rng(SEED))
    vrows = vdjdb_rows(vrec, veps, rng)

    assert not (set(veps) & set(imm_peps)), "panel epitopes overlap between datasets"

    # VDJdb receptors to embed, one row per distinct paired receptor
    vuniq = (vrec[["cdr3a", "va", "cdr3b", "vb"]].drop_duplicates().reset_index(drop=True)
             .rename(columns={"cdr3a": "CDR3A", "va": "TRAV", "cdr3b": "CDR3B", "vb": "TRBV"}))
    vuniq["key"] = vuniq.CDR3A + "|" + vuniq.CDR3B

    banned_a = set(pd.read_csv(BENCH, sep="\t").tcra_cdr3) | set(vuniq.CDR3A)
    banned_b = set(pd.read_csv(BENCH, sep="\t").tcrb_cdr3) | set(vuniq.CDR3B)
    obg = olga_background(banned_a, banned_b, rng)

    banned_pep = set(pd.read_csv(BENCH, sep="\t").peptide) | set(veps)
    for extra in ("iedb_positives.csv", "vdjdb_positives.csv"):     # the Kaggle-given positives
        p = os.path.join(REPO, "dump", "immrep25", extra)
        if os.path.exists(p):
            c = pd.read_csv(p)
            col = next((x for x in ("Peptide", "peptide", "epitope") if x in c.columns), None)
            if col:
                banned_pep |= set(c[col].astype(str))
    pbg = iedb_background(banned_pep, rng)

    assert not (set(obg.CDR3A) & banned_a) and not (set(obg.CDR3B) & banned_b), \
        "background receptors contain a benchmark CDR3"
    assert not (set(pbg.peptide) & banned_pep), "background peptides contain a benchmark peptide"

    vuniq.to_csv(os.path.join(WORK, "rec_vdjdb.csv"), index=False)
    obg.to_csv(os.path.join(WORK, "rec_olgabg.csv"), index=False)
    pd.DataFrame({"peptide": sorted(set(veps) | set(imm_peps))}).to_csv(
        os.path.join(WORK, "pep_panel.csv"), index=False)
    pbg.to_csv(os.path.join(WORK, "pep_bg.csv"), index=False)
    vrows.to_csv(os.path.join(WORK, "rows_vdjdb.csv"), index=False)
    imm[["peptide", "key", "kidx", "label"]].to_csv(os.path.join(WORK, "rows_immrep.csv"),
                                                    index=False)

    print("VDJdb panel (%s, HQ, >=%d pairs): %s" % (ALLELE, MIN_PAIRS, ", ".join(veps)))
    print("  %d distinct paired receptors -> %d rows, positive fraction %.4f"
          % (len(vuniq), len(vrows), vrows.label.mean()))
    print("IMMREP25 panel: %s" % ", ".join(imm_peps))
    print("  %d receptors -> %d rows, positive fraction %.4f"
          % (imm.key.nunique(), len(imm), imm.label.mean()))
    print("background: %d OLGA paired receptors, %d IEDB %s 9-mers (banned %d peptides)"
          % (len(obg), len(pbg), ALLELE, len(banned_pep)))
    print("wrote %s/{rec_vdjdb,rec_olgabg,pep_panel,pep_bg,rows_vdjdb,rows_immrep}.csv" % WORK)


# --------------------------------------------------------------------------- #
# basis, design, fits
# --------------------------------------------------------------------------- #
def fit_basis(X: np.ndarray, n: int, tag: str) -> dict:
    """PCA fitted on BACKGROUND only and persisted, esm_pca.py's pattern."""
    from sklearn.decomposition import PCA
    n = min(n, X.shape[1], X.shape[0])
    p = PCA(n_components=n, random_state=SEED).fit(X)
    out = os.path.join(WORK, "basis_%s.npz" % tag)
    np.savez(out, mean=p.mean_, components=p.components_, evr=p.explained_variance_ratio_,
             n=X.shape[0])
    return {"mean": p.mean_, "components": p.components_,
            "evr": float(p.explained_variance_ratio_.sum()), "k": n}


def apply_basis(X: np.ndarray, b: dict) -> np.ndarray:
    return (X - b["mean"]) @ b["components"].T


def design(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """[receptor PCs | peptide PCs | products]. The products are the ONLY channel through which the
    peptide can change a within-peptide ranking -- see note 2."""
    prod = (R[:, :KR, None] * P[:, None, :KE]).reshape(len(R), -1)
    return np.hstack([R, P, prod])


def peptide_ci(y: np.ndarray, s: np.ndarray, rng) -> tuple[float, float, float]:
    """One peptide's McClish partial AUC with a recentred percentile 95% CI -- see note 4.

    Reuses epitope_free's vectorised pauc01/tie_groups (verified to 1e-9 against sklearn) and
    structure_signal._ci's recentring. Classes are resampled separately to hold n_pos and n_neg.
    """
    from epitope_free import pauc01, tie_groups
    yy = y.astype(float)
    order, starts = tie_groups(s)
    point = pauc01(yy, order, starts, MAX_FPR)
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.r_[rng.choice(ip, len(ip), replace=True),
                    rng.choice(ineg, len(ineg), replace=True)]
        o, st = tie_groups(s[idx])
        boot[b] = pauc01(yy[idx], o, st, MAX_FPR)
    dev = boot - boot.mean()
    return (float(point), float(point + np.percentile(dev, 2.5)),
            float(point + np.percentile(dev, 97.5)))


def gain_aic(ytr: np.ndarray, yte: np.ndarray, pte: np.ndarray, k: int) -> dict:
    """G = 1 - CE/CE_chance (panel2_ce.py) and dAIC against the intercept-only model.

    CE_chance uses the TRAINING label marginal evaluated on the test labels, so a model that only
    reproduces the prior scores G = 0. Both datasets are held at the same positive fraction, so
    CE_chance is a shared constant and G is comparable across them.
    """
    eps = 1e-12
    p1 = float(ytr.mean())
    pri = np.where(yte == 1, p1, 1.0 - p1)
    ce_ch = float(-np.mean(np.log(np.clip(pri, eps, 1))))
    q = np.clip(np.where(yte == 1, pte, 1.0 - pte), eps, 1)
    ce = float(-np.mean(np.log(q)))
    n = len(yte)
    return {"ce": ce, "ce_chance": ce_ch, "G": 1.0 - ce / ce_ch,
            "dAIC": 2 * k - 2 * (-ce * n - (-ce_ch * n)), "nats": (ce_ch - ce) * n}


def arms(X: np.ndarray, y: np.ndarray, pep: np.ndarray, rec: np.ndarray, rng,
         collect: list | None = None) -> list[dict]:
    """Epitope seen in training (grouped 5-fold on receptor identity) and held out (strict LOPO)."""
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    proto = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    k = X.shape[1] + 1
    out = []
    for arm in ("seen", "unseen"):
        if arm == "seen":
            splits = list(GroupKFold(n_splits=5).split(X, y, rec))
        else:
            splits = []
            for p in np.unique(pep):
                te = pep == p
                cog = np.unique(rec[te & (y == 1)])
                splits.append((np.flatnonzero(~te & ~np.isin(rec, cog)), np.flatnonzero(te)))
        s = np.zeros(len(y))
        scored = np.zeros(len(y), dtype=bool)
        for tr, te in splits:
            m = clone(proto).fit(X[tr], y[tr])
            s[te] = m.predict_proba(X[te])[:, 1]
            scored[te] = True
        assert scored.all(), "%d rows never scored out-of-fold" % int((~scored).sum())

        # Per-epitope AUC with its CI, so (a) the exceedance count can use the author's CI rule and
        # (b) a depth confound can be separated from epitope-specific signal: VDJdb's panel carries
        # 175 positives per epitope except CLGGLLTMV (100) and FLYALALLL (78), and if only those
        # fell below the leader the aggregate would be a size effect.
        rows = []
        for p in np.unique(pep):
            m = pep == p
            if not (0 < y[m].sum() < m.sum()):
                continue
            point, lo, hi = peptide_ci(y[m], s[m], rng)
            rows.append(dict(arm=arm, peptide=p, n=int(m.sum()), n_pos=int(y[m].sum()),
                             auc01=point, ci_lo=lo, ci_hi=hi,
                             clears=bool(lo > AUC_BEST_FULL)))
        a = np.array([r["auc01"] for r in rows])
        # G and dAIC are computed ONCE on the pooled out-of-fold predictions, exactly as the macro
        # is. Summing per-fold dAIC charges 2k for EVERY fold -- 5 x 161 parameters for a single
        # 161-parameter model -- which made the seen values report dAIC > 0 while G > 0, i.e. the
        # two read-outs contradicted each other. Pooling is exact rather than approximate here
        # because every fold carries the same 1:9 prior by construction, so CE_chance is
        # fold-invariant.
        out.append(dict(arm=arm, k=k, n_ep=len(rows), macro01=float(a.mean()),
                        max01=float(a.max()), median01=float(np.median(a)),
                        n_clear=int(sum(r["clears"] for r in rows)),
                        **gain_aic(y, y, s, k)))
        if collect is not None:
            collect += rows
    return out


def load_emb(tag: str) -> np.ndarray:
    return np.load(os.path.join(WORK, "emb_%s.npy" % tag))


def guard(X: np.ndarray, code: np.ndarray, tag: str):
    """The pinning guard, IN THE MAIN PATH -- see note 2."""
    nun = len(np.unique(np.round(X, 6), axis=0))
    nrec = len(np.unique(code))
    assert nun > nrec, ("%s: the design holds %d distinct rows for %d receptors, so the peptide "
                        "side is absent and every macro would be exactly 0.5 by construction"
                        % (tag, nun, nrec))
    assert np.isfinite(X).all(), "%s: non-finite design" % tag


def score():
    assert os.path.isdir(WORK), "run --prep and learnability_embed.py first (LEARNABILITY_WORK=%s)" \
        % WORK
    rows_v = pd.read_csv(os.path.join(WORK, "rows_vdjdb.csv"))
    rows_i = pd.read_csv(os.path.join(WORK, "rows_immrep.csv"))
    rec_v = pd.read_csv(os.path.join(WORK, "rec_vdjdb.csv"))
    pep_panel = pd.read_csv(os.path.join(WORK, "pep_panel.csv"))

    # peptide basis, shared by both datasets
    Pbg = load_emb("pep_bg_esm")
    pb = fit_basis(Pbg, P_PC, "pep_esm")
    Ppan = apply_basis(load_emb("pep_panel_esm"), pb)
    pidx = {p: i for i, p in enumerate(pep_panel.peptide)}

    res, per, evr = [], [], {"pep": pb["evr"]}
    for repname in ("sceptr", "esm"):
        rb = fit_basis(load_emb("rec_olgabg_%s" % repname), R_PC, "rec_%s" % repname)
        evr[repname] = rb["evr"]

        # --- VDJdb ---
        Rv = apply_basis(load_emb("rec_vdjdb_%s" % repname), rb)
        ridx = {k: i for i, k in enumerate(rec_v.key)}
        R = Rv[[ridx[k] for k in rows_v.key]]
        P = Ppan[[pidx[p] for p in rows_v.peptide]]
        X = design(R, P)
        _, code = np.unique(rows_v.key.to_numpy(), return_inverse=True)
        guard(X, code, "vdjdb/%s" % repname)
        pe = []
        for r in arms(X, rows_v.label.to_numpy(), rows_v.peptide.to_numpy(), code,
                      np.random.default_rng(SEED), pe):
            res.append(dict(dataset="vdjdb", rep=repname, n_rows=len(X), **r))
        per += [dict(dataset="vdjdb", rep=repname, **e) for e in pe]

        # --- IMMREP25: reuse the committed cache, subset to this allele's rows ---
        Ri = apply_basis(np.load(os.path.join(
            CACHE_EMBED, "sceptr_default.npy" if repname == "sceptr" else "esm2_35M.npy")), rb)
        R = Ri[rows_i.kidx.to_numpy()]
        P = Ppan[[pidx[p] for p in rows_i.peptide]]
        X = design(R, P)
        _, code = np.unique(rows_i.key.to_numpy(), return_inverse=True)
        guard(X, code, "immrep/%s" % repname)
        pe = []
        for r in arms(X, rows_i.label.to_numpy(), rows_i.peptide.to_numpy(), code,
                      np.random.default_rng(SEED), pe):
            res.append(dict(dataset="immrep25", rep=repname, n_rows=len(X), **r))
        per += [dict(dataset="immrep25", rep=repname, **e) for e in pe]

    t = pd.DataFrame(res)
    t["n_rec"] = t.dataset.map({"vdjdb": rows_v.key.nunique(), "immrep25": rows_i.key.nunique()})
    for tag, v in evr.items():
        t["evr_%s" % tag] = v
    t.to_csv(os.path.join(RESULTS, "learnability.csv"), index=False)
    pf = pd.DataFrame(per)
    pf.to_csv(os.path.join(RESULTS, "learnability_per_epitope.csv"), index=False)

    print("\n=== binder/non-binder learnability, %s, matched 1:9 geometry ===" % ALLELE)
    print("floor 0.4737 | peptide-blind cap 0.5322 | leader %.6f | pwCv 0.659\n" % AUC_BEST_FULL)
    print(t[["dataset", "rep", "arm", "n_rows", "n_ep", "k", "macro01", "max01",
             "n_clear", "G", "nats", "dAIC"]]
          .to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nG>0 = the representation pays information rent; dAIC<0 = it earns its parameters.")
    print("n_clear counts peptides whose 95%% CI LOWER bound exceeds the leader %.6f (B=%d)."
          % (AUC_BEST_FULL, N_BOOT))
    print("\nwrote results/learnability.csv + results/learnability_per_epitope.csv")
    emit_macros(t, pf)


# --------------------------------------------------------------------------- #
# macros and the supplementary table
# --------------------------------------------------------------------------- #
def _cell(t: pd.DataFrame, ds: str, rep: str, arm: str, col: str):
    return t[(t.dataset == ds) & (t.rep == rep) & (t.arm == arm)][col].iloc[0]


def emit_macros(t: pd.DataFrame, pf: pd.DataFrame) -> dict:
    """Macros from the scored tables.

    Split from score() deliberately, as embedding_comparison.emit_macros is: the read-outs are a
    pure function of the two CSVs, so revising which numbers the paper names must not cost a refit.
    `python src/learnability.py --macros` re-emits from the committed CSVs.

    Nothing here re-emits a published quantity: the leaderboard best is \\scgBestFull, the
    peptide-blind cap \\scgBound, the McClish floor \\blindFloor, the pairwise ceiling \\pwCv.
    """
    from scipy.stats import spearmanr

    g = lambda ds, rep, arm, col: _cell(t, ds, rep, arm, col)      # noqa: E731
    m = {
        "lrnAllele": ALLELE,
        "lrnNepi": "%d" % int(t.n_ep.iloc[0]),
        "lrnNrowsV": "%d" % int(g("vdjdb", "sceptr", "seen", "n_rows")),
        "lrnNrowsI": "%d" % int(g("immrep25", "sceptr", "seen", "n_rows")),
        "lrnNrecV": "%d" % int(g("vdjdb", "sceptr", "seen", "n_rec")),
        "lrnNrecI": "%d" % int(g("immrep25", "sceptr", "seen", "n_rec")),
        "lrnK": "%d" % int(t.k.iloc[0]),
        "lrnTwoK": "%d" % (2 * int(t.k.iloc[0])),
        "lrnRpc": "%d" % R_PC,
        "lrnPpc": "%d" % P_PC,
        "lrnNprod": "%d" % (KR * KE),
        "lrnPosFrac": "0.100",
        "lrnNbg": "%d" % N_BG,
        "lrnNboot": "%d" % N_BOOT,
        "lrnCeChance": "%.4f" % t.ce_chance.iloc[0],
        "lrnEvrRecS": "%.1f" % (100 * t.evr_sceptr.iloc[0]),
        "lrnEvrRecE": "%.1f" % (100 * t.evr_esm.iloc[0]),
        "lrnEvrPep": "%.1f" % (100 * t.evr_pep.iloc[0]),
    }
    # the two representations, both datasets, both conditions
    for rep, rtag in (("sceptr", ""), ("esm", "E")):
        for ds, dtag in (("vdjdb", "Vdj"), ("immrep25", "Imm")):
            for arm, atag in (("seen", ""), ("unseen", "U")):
                m["lrn%sMacro%s%s" % (dtag, atag, rtag)] = "%.4f" % g(ds, rep, arm, "macro01")
                m["lrn%sMax%s%s" % (dtag, atag, rtag)] = "%.4f" % g(ds, rep, arm, "max01")
                m["lrn%sNclear%s%s" % (dtag, atag, rtag)] = "%d" % int(g(ds, rep, arm, "n_clear"))
                m["lrn%sG%s%s" % (dtag, atag, rtag)] = "%.4f" % g(ds, rep, arm, "G")
                m["lrn%sAic%s%s" % (dtag, atag, rtag)] = "%.0f" % g(ds, rep, arm, "dAIC")
    # the deviance decomposition -- the sharp form of the AIC result. Two scales, so two formats.
    m["lrnVdjNats"] = "%.0f" % g("vdjdb", "sceptr", "seen", "nats")
    m["lrnVdjNatsE"] = "%.0f" % g("vdjdb", "esm", "seen", "nats")
    m["lrnImmNats"] = "%.2f" % g("immrep25", "sceptr", "seen", "nats")
    m["lrnImmNatsE"] = "%.2f" % g("immrep25", "esm", "seen", "nats")
    m["lrnVdjBits"] = "%.3f" % (g("vdjdb", "sceptr", "seen", "nats")
                                / g("vdjdb", "sceptr", "seen", "n_rows") / np.log(2))
    m["lrnVdjBitsE"] = "%.3f" % (g("vdjdb", "esm", "seen", "nats")
                                 / g("vdjdb", "esm", "seen", "n_rows") / np.log(2))

    # the depth confound, SCEPTR, epitope seen: reported with its P and its tie structure because
    # neither representation's rank correlation reaches 0.05 -- the extremes carry the claim.
    v = pf[(pf.dataset == "vdjdb") & (pf.rep == "sceptr") & (pf.arm == "seen")]
    rho, pval = spearmanr(v.n_pos, v.auc01)
    lo, hi = v.loc[v.n_pos.idxmin()], v.loc[v.auc01.idxmin()]
    m.update({
        "lrnRho": "%.3f" % rho,
        "lrnRhoP": "%.2f" % pval,
        "lrnNtied": "%d" % int((v.n_pos == v.n_pos.max()).sum()),
        "lrnNtiedPos": "%d" % int(v.n_pos.max()),
        "lrnShallowPep": str(lo.peptide),
        "lrnShallowN": "%d" % int(lo.n_pos),
        "lrnShallowAuc": "%.4f" % lo.auc01,
        "lrnDeepPep": str(hi.peptide),
        "lrnDeepN": "%d" % int(hi.n_pos),
        "lrnDeepAuc": "%.4f" % hi.auc01,
    })
    ve = pf[(pf.dataset == "vdjdb") & (pf.rep == "esm") & (pf.arm == "seen")]
    rhoe, pvale = spearmanr(ve.n_pos, ve.auc01)
    m["lrnRhoE"] = "%.3f" % rhoe
    m["lrnRhoPE"] = "%.2f" % pvale

    path = os.path.join(ADAT, "learnability_macros.tex")
    with open(path, "w") as fh:
        for k, v_ in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v_))
    print("wrote %s (%d macros)" % (path, len(m)))
    write_table(t)
    return m


def _tex(v: str) -> str:
    """A leading ASCII minus becomes a real math minus, as the other generated tables do."""
    return ("$-$" + v[1:]) if v.startswith("-") else v


def write_table(t: pd.DataFrame):
    """The supplementary table: a bare `tabular` plus the two-line provenance banner.

    House pattern is tables/mi_master_table.tex -- the supplement owns the float, the caption and
    the \\label. Written through paths.TABLES so a worktree session lands it where it can see it.
    """
    order = [(ds, rep, arm) for ds in ("vdjdb", "immrep25")
             for rep in ("sceptr", "esm") for arm in ("seen", "unseen")]
    name = {"vdjdb": "VDJdb$^{(HQ)}$", "immrep25": "IMMREP25$^{(+)}$"}
    best = {"macro01": t.macro01.max(), "n_clear": t.n_clear.max(),
            "G": t.G.max(), "dAIC": t.dAIC.min()}

    os.makedirs(paths.TABLES, exist_ok=True)
    out = os.path.join(paths.TABLES, "learnability_table.tex")
    with open(out, "w") as fh:
        fh.write("%% GENERATED by src/learnability.py -- do not edit.\n"
                 "%% Regenerate: python src/learnability.py --macros"
                 " (from the 2026-immrep25-audit repo).\n")
        fh.write(r"\begin{tabular}{lllrrrrr}" + "\n")
        fh.write(r"\toprule" + "\n")
        fh.write("dataset & representation & epitope & rows & macro-AUC$_{0.1}$ & "
                 "peptides clearing leader & $G$ & $\\Delta$AIC \\\\\n")
        fh.write(r"\midrule" + "\n")
        for i, (ds, rep, arm) in enumerate(order):
            if i == 4:
                fh.write(r"\midrule" + "\n")
            cells = []
            for col, fmt in (("macro01", "%.4f"), ("n_clear", "%d"),
                             ("G", "%.4f"), ("dAIC", "%.0f")):
                v = _cell(t, ds, rep, arm, col)
                s = _tex(fmt % v)
                cells.append(r"\textbf{%s}" % s if v == best[col] else s)
            fh.write("%s & %s & %s & %d & %s & %s & %s & %s \\\\\n"
                     % (name[ds], REP_LABEL[rep], ARM_LABEL[arm],
                        int(_cell(t, ds, rep, arm, "n_rows")), *cells))
        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")
    print("wrote %s" % out)


# --------------------------------------------------------------------------- #
def demo():
    """Self-checks that do not need the embeddings: allele assignment, geometry, the read-outs."""
    assert _norm_allele("HLA-A*02:01,HLA-A*02") == "A*02:01"
    assert _norm_allele("HLA-A*02") == "A*02:01"           # merged, see note 1
    assert _norm_allele("HLA-A*02:01:48") == "A*02:01"
    assert _norm_allele("HLA-B*08:01") == "B*08:01"

    al = epitope_alleles()
    for ep, want in (("FLRGRAYGL", "B*08:01"), ("QAKWRLQTL", "B*08:01"),
                     ("RPPIFIRRL", "B*07:02"), ("NLVPMVATV", "A*02:01"),
                     ("GILGFVFTL", "A*02:01")):
        got = al.loc[ep, "allele"]
        assert got == want, "%s -> %s, expected %s" % (ep, got, want)

    imm, _, peps = immrep_panel()
    assert len(peps) == PANEL and abs(imm.label.mean() - 0.1) < 1e-9

    # the matched-geometry check: G is only comparable across datasets if CE_chance is, and
    # CE_chance is a function of the positive fraction. Per peptide, not just pooled.
    vrec, veps = vdjdb_panel(np.random.default_rng(SEED))
    vrows = vdjdb_rows(vrec, veps, np.random.default_rng(SEED))
    assert abs(vrows.label.mean() - imm.label.mean()) < 1e-9, \
        "positive fractions differ: vdjdb %.6f vs immrep %.6f" % (vrows.label.mean(),
                                                                  imm.label.mean())
    for frame, name in ((vrows, "vdjdb"), (imm, "immrep")):
        f = frame.groupby("peptide").label.mean()
        assert (f - 0.1).abs().max() < 1e-9, "%s per-peptide ratio is ragged: %s" % (name, dict(f))
    assert vrows.groupby(["peptide", "key"]).label.nunique().max() == 1, "a cell carries both labels"

    # G and dAIC must agree in sign on planted signal, and read ~0 on none
    rng = np.random.default_rng(SEED)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    good = np.clip(np.where(y == 1, 0.35, 0.07) + rng.normal(0, .02, n), 1e-3, 1 - 1e-3)
    flat = np.full(n, 0.1)
    g1 = gain_aic(y, y, good, k=10)
    g0 = gain_aic(y, y, flat, k=10)
    assert g1["G"] > 0.1 and g1["dAIC"] < 0, g1
    assert abs(g0["G"]) < 1e-3 and g0["dAIC"] > 0, g0

    # a receptor-only design must be pinned: full AUC exactly 0.5 per peptide
    from sklearn.metrics import roc_auc_score
    pep = imm.peptide.to_numpy()
    _, code = np.unique(imm.key.to_numpy(), return_inverse=True)
    s = rng.normal(size=code.max() + 1)[code]           # one score per receptor, no peptide
    full = np.mean([roc_auc_score((imm.label.to_numpy())[pep == p], s[pep == p])
                    for p in np.unique(pep)])
    assert abs(full - 0.5) < 1e-9, full

    # NEW 1: the reused partial AUC must agree with sklearn on a real per-peptide task, since the
    # CI rule now rests on pauc01 rather than on roc_auc_score
    from epitope_free import pauc01, tie_groups
    y1 = imm.label.to_numpy()[pep == peps[0]]
    s1 = s[pep == peps[0]]
    o, st = tie_groups(s1)
    assert abs(pauc01(y1.astype(float), o, st, MAX_FPR)
               - roc_auc_score(y1, s1, max_fpr=MAX_FPR)) < 1e-9, "pauc01 disagrees with sklearn"

    # NEW 2: the bootstrap must bracket its own point estimate, and must hold the class counts
    point, lo, hi = peptide_ci(y1, s1, np.random.default_rng(SEED))
    assert lo <= point <= hi, (lo, point, hi)
    assert lo < AUC_BEST_FULL, "a random receptor-only score cleared the leader: %.4f" % lo

    print("learnability.demo OK")
    print("  modal-allele rule ejects FLRGRAYGL/QAKWRLQTL (B*08:01) and RPPIFIRRL (B*07:02)")
    print("  IMMREP25 %s: %d peptides, %d rows, positive fraction %.3f"
          % (ALLELE, len(peps), len(imm), imm.label.mean()))
    print("  planted signal: G=%+.3f dAIC=%+.0f | no signal: G=%+.4f dAIC=%+.0f"
          % (g1["G"], g1["dAIC"], g0["G"], g0["dAIC"]))
    print("  receptor-only macro FULL AUC = %.12f (the rank identity, to 1e-9)" % full)
    print("  pauc01 == sklearn to 1e-9; bootstrap CI %.4f--%.4f brackets %.4f (B=%d)"
          % (lo, hi, point, N_BOOT))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--prep" in sys.argv:
        prep()
    elif "--score" in sys.argv:
        score()
    elif "--macros" in sys.argv:            # re-emit from the committed CSVs, no refitting
        emit_macros(pd.read_csv(os.path.join(RESULTS, "learnability.csv")),
                    pd.read_csv(os.path.join(RESULTS, "learnability_per_epitope.csv")))
    else:
        print("use --prep | --score | --macros | --demo")
