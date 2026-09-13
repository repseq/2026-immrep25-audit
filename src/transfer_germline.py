#!/usr/bin/env python
# 2026-09-13  Do germline-only classifiers trained on VDJdb transfer into IMMREP25?
#
# This is the condition three reviewers asked for (R1.1, R2.4, R3.5). The manuscript's germline
# result is cross-validated WITHIN IMMREP25, so it is an in-distribution ceiling and cannot be
# set against the challenge's blind 0.60. The repo's other blind arm, germline_baseline.lopo_unseen,
# is blind to the PEPTIDE but not to the RECEPTORS: under leave-one-peptide-out a held-out
# peptide's own receptors still occur in training, labelled negative nine times each, while the
# receptors they must be ranked against occur labelled positive -- so the model ranks them
# backwards and the full AUC lands below chance for a reason that is an artefact of receptor
# reuse, not a statement about germline.
#
# Training on an EXTERNAL resource removes that artefact: VDJdb carries 0 records on any of
# IMMREP25's 20 peptides (verified below), and only 2.5% of IMMREP25's TCRbeta clonotypes occur
# in VDJdb-HQ at all. So this is the entrants' actual condition -- fit on peptides you have,
# score peptides you have never seen -- with no CDR3 motif anywhere in the feature set.
#
# The two datasets use DISJOINT gene vocabularies (IMMREP25: Adaptive `TCRBV03-01/03-02`;
# VDJdb: IMGT `TRBV3-1`), so without harmonisation every test gene encodes as unseen, the
# receptor channel vanishes, and the peptide features are constant within a peptide -- yielding
# a null that measures the encoding, not the biology. Gene names are therefore mapped through
# our own VDJtools CDR-validated table (see SOURCES.md); the mapped-gene coverage is asserted,
# not assumed.
#
# Arms:
#   internal  leave-one-epitope-out WITHIN VDJdb -- the probe-power control. If germline+peptide
#             features cannot transfer to an unseen epitope even inside VDJdb, a null on
#             IMMREP25 says nothing about IMMREP25.
#   transfer  train on all qualifying VDJdb epitopes, score IMMREP25's 20 peptides.
#   matched   train on A*02:01 VDJdb epitopes only, score IMMREP25's 10 A*02:01 peptides.
#             B*40:01 has no matched arm: VDJdb holds 2 records over 1 nine-mer B*40:01 epitope.
#
# Run: python src/transfer_germline.py      (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDJDB = os.path.join(REPO, "dump", "vdjdb-2026-06-03", "vdjdb.slim.txt")
IMMREP = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
GENE_MAP = os.environ.get(
    "VDJTOOLS_ADAPTIVE_MAP",
    os.path.expanduser("~/vcs/code/vdjtools/python/vdjtools/resources/adaptive_imgt_map.tsv"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

AA = list("ACDEFGHIKLMNPQRSTVWY")
MIN_REC = 40        # receptors an epitope needs in total to qualify as a training epitope
MIN_HALF = 10       # ... and in each of the two receptor-disjoint halves
CAP = 200           # receptors kept per epitope; VDJdb is skewed (median 2, max 29,698) and one
                    # epitope would otherwise supply a third of all training rows
PANEL = 10          # epitopes per mimicked panel -- IMMREP25 uses ten per allele
NEG = PANEL - 1     # negatives per positive, as IMMREP25 forms them
N_HELDOUT = 20      # held-out epitopes for the internal arm, matching IMMREP25's twenty
MAX_ITER = 200
SEED = 0
MAX_FPR = 0.1
COMPETITOR = 0.60   # best of 126 IMMREP25 submissions, macro AUC_0.1 (Richardson 2026)
P01_FLOOR = 0.5 * (1 - (MAX_FPR / 2) / (1 - MAX_FPR / 2))    # 9/19, the metric's floor


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _canon(col: pl.Expr) -> pl.Expr:
    """VDJdb ships comma-separated ambiguity lists with alleles: take the first gene, drop the
    allele. Mirrors load_data.canon_gene, which does the same for the cohort loaders."""
    return col.cast(pl.Utf8).str.split(",").list.first().str.split("*").list.first()


def _allele(col: pl.Expr) -> pl.Expr:
    """`HLA-A*02:01,HLA-A*02` -> `A*02:01`: first entry, HLA- prefix dropped, two fields kept."""
    first = col.cast(pl.Utf8).str.split(",").list.first().str.strip_prefix("HLA-")
    return (first.str.split(":").list.slice(0, 2).list.join(":"))


def gene_map() -> dict[str, str]:
    """Adaptive token -> IMGT gene, from our own VDJtools CDR-validated table."""
    m = pl.read_csv(GENE_MAP, separator="\t")
    return dict(zip(m["adaptive_token"].to_list(), m["imgt_gene"].to_list()))


def vdjdb_receptors(chain: str) -> pl.DataFrame:
    """Distinct 9-mer-restricted human MHC-I receptors of one chain: epitope, allele, V, J, len."""
    v = (pl.read_csv(VDJDB, separator="\t", infer_schema_length=10000)
         .filter((pl.col("species") == "HomoSapiens") & (pl.col("mhc.class") == "MHCI")
                 & (pl.col("gene") == chain))
         .with_columns(peptide=pl.col("antigen.epitope").cast(pl.Utf8),
                       cdr3=pl.col("cdr3").cast(pl.Utf8),
                       v=_canon(pl.col("v.segm")), j=_canon(pl.col("j.segm")),
                       allele=_allele(pl.col("mhc.a")))
         .filter(pl.col("peptide").str.len_chars() == 9)
         .filter(pl.col("cdr3").str.len_chars() >= 5)
         # polars' unique() neither preserves order nor fixes WHICH duplicate it keeps, and a
         # CDR3 reported by several studies carries several V/J calls -- so without sorting
         # first, both the retained gene call and the row order change between runs, and every
         # downstream capped sample changes with them. Sorting makes the representative and the
         # order a function of the data.
         .sort("peptide", "cdr3", "v", "j")
         .unique(["peptide", "cdr3"], keep="first", maintain_order=True)
         .select("peptide", "allele", "cdr3", "v", "j"))
    return v.with_columns(cdr3_len=pl.col("cdr3").str.len_chars())


def immrep_receptors(chain: str) -> pl.DataFrame:
    """IMMREP25's 10,000 labelled pairs for one chain, genes harmonised to IMGT."""
    pre = "tcrb" if chain == "TRB" else "tcra"
    mp = gene_map()
    d = (pl.read_csv(IMMREP, separator="\t")
         .select(peptide=pl.col("peptide").cast(pl.Utf8),
                 allele=pl.col("hla").cast(pl.Utf8),
                 cdr3=pl.col("%s_cdr3" % pre).cast(pl.Utf8),
                 v_raw=pl.col("%s_v" % pre).cast(pl.Utf8),
                 j_raw=pl.col("%s_j" % pre).cast(pl.Utf8),
                 label=pl.col("label").cast(pl.Int64))
         .with_columns(v=pl.col("v_raw").replace_strict(mp, default=None),
                       j=pl.col("j_raw").replace_strict(mp, default=None),
                       cdr3_len=pl.col("cdr3").str.len_chars()))
    unmapped = d.filter(pl.col("v").is_null() | pl.col("j").is_null()).height
    assert unmapped == 0, "%d IMMREP25 rows have an unmapped gene name" % unmapped
    return d.drop("v_raw", "j_raw")


def qualifying(rec: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Qualifying epitopes, capped, and split into two receptor-disjoint halves.

    The half is a deterministic function of the CDR3 itself rather than of the row, so a
    receptor occurring under two epitopes lands in the same half both times. That is what makes
    the internal arm genuinely receptor-disjoint instead of merely row-disjoint -- without it,
    the arm would inherit the same receptor-reuse inversion that makes leave-one-peptide-out
    within IMMREP25 read below chance, and would understate the probe's power.
    """
    rec = rec.with_columns(half=(pl.col("cdr3").hash(seed=SEED) % 2).cast(pl.Int64))
    tot = rec.group_by("peptide").agg(n=pl.len())
    per_half = (rec.group_by("peptide", "half").agg(n=pl.len())
                .group_by("peptide").agg(n_min=pl.col("n").min(), n_halves=pl.len()))
    keep = (tot.join(per_half, on="peptide")
            .filter((pl.col("n") >= MIN_REC) & (pl.col("n_halves") == 2)
                    & (pl.col("n_min") >= MIN_HALF))["peptide"].to_list())
    out = []
    for ep in sorted(keep):
        for h in (0, 1):
            sub = (rec.filter((pl.col("peptide") == ep) & (pl.col("half") == h))
                   .sort("cdr3"))
            if sub.height > CAP // 2:
                sub = sub[np.sort(rng.choice(sub.height, CAP // 2, replace=False))]
            out.append(sub)
    return pl.concat(out)


def panels(rec: pl.DataFrame, rng: np.random.Generator) -> dict[str, list[str]]:
    """For each epitope, up to NEG partner epitopes OF THE SAME ALLELE -- the set its receptors
    are re-paired against, mimicking IMMREP25's within-MHC negative construction. Alleles with
    too few qualifying epitopes fall back to any other epitope, which is recorded by the caller."""
    # VDJdb reports some epitopes under more than one allele; take the lexicographic minimum so
    # the same-allele partition is a function of the data rather than of row order
    ep_allele = {r["peptide"]: r["allele"] for r in
                 rec.group_by("peptide").agg(allele=pl.col("allele").min())
                 .sort("peptide").to_dicts()}
    by_allele: dict[str, list[str]] = {}
    for ep, al in ep_allele.items():
        by_allele.setdefault(al, []).append(ep)
    all_eps = sorted(ep_allele)
    out = {}
    for ep in all_eps:
        same = sorted(set(by_allele[ep_allele[ep]]) - {ep})
        pool = same if len(same) >= NEG else sorted(set(all_eps) - {ep})
        take = min(NEG, len(pool))
        out[ep] = sorted(rng.choice(pool, take, replace=False).tolist()) if take else []
    return out


def task_rows(eps: list[str], rec: pl.DataFrame, pan: dict[str, list[str]]) -> pl.DataFrame:
    """IMMREP25's row construction: every receptor of epitope e is a positive for e and a
    negative for each of e's panel partners. One row per (receptor, peptide)."""
    # by_ep must cover the partners too, not just `eps` -- otherwise a single-epitope call
    # (the internal arm's test task) drops every negative and yields positives only.
    need = set(eps) | {o for ep in eps for o in pan.get(ep, [])}
    by_ep = {e: f for e in sorted(need)
             if (f := rec.filter(pl.col("peptide") == e)).height}
    frames = []
    for ep in eps:
        if ep not in by_ep:
            continue
        own = by_ep[ep].with_columns(label=pl.lit(1, pl.Int64))
        frames.append(own.select("peptide", "allele", "cdr3", "v", "j", "cdr3_len", "label"))
        for other in pan[ep]:
            if other not in by_ep:
                continue
            frames.append(by_ep[other]
                          .with_columns(peptide=pl.lit(ep), allele=pl.lit(by_ep[ep]["allele"][0]),
                                        label=pl.lit(0, pl.Int64))
                          .select("peptide", "allele", "cdr3", "v", "j", "cdr3_len", "label"))
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# features and model
# --------------------------------------------------------------------------- #
def peptide_features(peps: list[str]) -> pl.DataFrame:
    """Amino-acid composition of each distinct peptide. Length is omitted: the panel and the
    training set are all 9-mers, so it is constant and carries no information."""
    uniq = sorted(set(peps))
    comp = {"peptide": uniq}
    for a in AA:
        comp["p_%s" % a] = [p.count(a) / len(p) for p in uniq]
    return pl.DataFrame(comp)


# `allele` is deliberately NOT a feature: negatives are constructed within allele, the test set
# carries only two, and peptide composition already encodes the allele's anchor preference (every
# B*40:01 peptide in this panel carries Glu at P2). It would add nothing and would put VDJdb's
# allele vocabulary against sklearn's 255-category ceiling.
CAT = ["v", "j"]
NUM = ["cdr3_len"] + ["p_%s" % a for a in AA]


def _design(df: pl.DataFrame, cats: dict[str, list[str]] | None):
    """Numpy design matrix with categoricals integer-coded against a FIXED vocabulary.

    polars -> numpy directly: sklearn's native interface, and it avoids the polars -> pandas
    hop (which would pull in pyarrow for nothing). Categories are fixed from the training frame;
    a test value outside them becomes NaN, which HistGradientBoosting treats as missing -- the
    honest encoding for an unseen gene. The assertion in immrep_receptors and the coverage
    figure printed by main() are what guarantee this path is not silently swallowing the whole
    receptor channel.
    """
    d = df.join(peptide_features(df["peptide"].to_list()), on="peptide", how="left")
    built, cols = {}, []
    for c in CAT:
        levels = cats[c] if cats is not None else sorted(set(d[c].drop_nulls().to_list()))
        built[c] = levels
        idx = {v: i for i, v in enumerate(levels)}
        cols.append(np.array([idx.get(v, np.nan) for v in d[c].to_list()], dtype=float))
    for c in NUM:
        cols.append(d[c].cast(pl.Float64).to_numpy())
    return np.column_stack(cols), d["label"].to_numpy(), built


def fit_predict(train: pl.DataFrame, test: pl.DataFrame):
    Xtr, ytr, cats = _design(train, None)
    Xte, yte, _ = _design(test, cats)
    m = HistGradientBoostingClassifier(categorical_features=list(range(len(CAT))),
                                       max_iter=MAX_ITER, learning_rate=0.05,
                                       random_state=SEED)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], yte, cats


def per_peptide(test: pl.DataFrame, score: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """One row per peptide: the benchmark's own read-out plus the full AUC."""
    peps = test["peptide"].to_list()
    alleles = test["allele"].to_list()
    rows = []
    for ep in sorted(set(peps)):
        mask = np.array([p == ep for p in peps])
        ys, ss = y[mask], score[mask]
        if ys.min() == ys.max():
            continue
        rows.append(dict(peptide=ep, allele=alleles[int(np.argmax(mask))],
                         n_pos=int(ys.sum()), n_neg=int((1 - ys).sum()),
                         auc01=float(roc_auc_score(ys, ss, max_fpr=MAX_FPR)),
                         auc=float(roc_auc_score(ys, ss))))
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def match_ratio(te: pl.DataFrame, rng: np.random.Generator,
                neg_per_pos: int = NEG) -> pl.DataFrame:
    """Subsample evaluation negatives to NEG per positive, matching IMMREP25's 50:450.

    Each partner epitope contributes all of its capped receptors, which leaves the internal
    arms at roughly 1:5 -- unmatched to the benchmark on exactly the axis the reviewers query,
    and the arms exist to be condition-matched. McClish standardisation is prevalence-
    independent, so this changes the estimate's variance rather than its target, but it makes
    the cross-dataset comparison like-for-like instead of merely defensible.
    """
    pos = te.filter(pl.col("label") == 1)
    neg = te.filter(pl.col("label") == 0)
    want = neg_per_pos * pos.height
    if neg.height > want:
        neg = neg[np.sort(rng.choice(neg.height, want, replace=False))]
    return pl.concat([pos, neg])


def arm_internal(rec: pl.DataFrame, pan: dict[str, list[str]],
                 rng: np.random.Generator, seen: bool = False) -> pl.DataFrame:
    """Probe power: leave-one-epitope-out inside VDJdb, peptide-unseen AND receptor-disjoint.

    Training uses half 0 with the held-out epitope removed outright; the test task is built
    from half 1, so neither the held-out epitope's own receptors nor the partner receptors it
    is ranked against were seen in training. This matches the transfer arm's condition, where
    IMMREP25's receptors are external to VDJdb.

    With `seen=True` the epitope stays in training and only the receptors are held out, which
    is the in-distribution condition -- the VDJdb analogue of the manuscript's within-IMMREP25
    cross-validated germline result. Comparing the two arms separates "this feature class
    carries no pool signal at all" from "it carries pool signal that does not cross epitopes",
    and that distinction is what decides whether the transfer arm's null is interpretable.
    """
    eps = sorted(pan)
    # sort on (count desc, peptide asc): almost every qualifying epitope sits exactly at the
    # per-half cap, so a sort on the count alone leaves a large tied block whose order polars
    # does not fix -- and the two arms would then hold out DIFFERENT epitopes, which destroys
    # the paired comparison they exist to support
    order = (rec.group_by("peptide").agg(n=pl.len())
             .sort(["n", "peptide"], descending=[True, False])["peptide"].to_list())
    held = [e for e in order if e in pan][:N_HELDOUT]
    out = []
    # a fresh generator, so both internal arms draw the SAME negative subsets and differ only
    # in whether the held-out epitope was in training
    sub_rng = np.random.default_rng(SEED)
    for ep in held:
        train_eps = eps if seen else [e for e in eps if e != ep]
        base = rec.filter(pl.col("half") == 0)
        tr = task_rows(train_eps, base if seen else base.filter(pl.col("peptide") != ep), pan)
        te = match_ratio(task_rows([ep], rec.filter(pl.col("half") == 1), pan), sub_rng)
        if te.height == 0 or int(te["label"].sum()) in (0, te.height):
            continue
        assert not set(tr["cdr3"]) & set(te["cdr3"]), "internal arm is not receptor-disjoint"
        s, y, _ = fit_predict(tr, te)
        r = per_peptide(te, s, y)
        if r.height:
            out.append(r.with_columns(arm=pl.lit("internal_seen" if seen else "internal")))
    return pl.concat(out) if out else pl.DataFrame()


def arm_transfer(rec: pl.DataFrame, pan: dict[str, list[str]], imm: pl.DataFrame,
                 label: str) -> pl.DataFrame:
    tr = task_rows(sorted(pan), rec, pan)
    te = imm.select("peptide", "allele", "cdr3", "v", "j", "cdr3_len", "label")
    s, y, cats = fit_predict(tr, te)
    seen = te.filter(pl.col("v").is_in(cats["v"]) & pl.col("j").is_in(cats["j"])).height
    cov = seen / te.height
    r = per_peptide(te, s, y).with_columns(arm=pl.lit(label))
    return r, cov, tr.height


def main():
    rng = np.random.default_rng(SEED)
    frames, cov_rows = [], []
    for chain in ("TRB", "TRA"):
        rec = vdjdb_receptors(chain)
        imm = immrep_receptors(chain)
        leak = rec.filter(pl.col("peptide").is_in(imm["peptide"].unique().to_list())).height
        assert leak == 0, "VDJdb carries %d records on IMMREP25 peptides -- not a blind test" % leak
        q = qualifying(rec, rng)
        pan = panels(q, rng)
        print("\n=== %s ===" % chain)
        print("VDJdb: %d distinct (epitope, CDR3) receptors over %d 9-mer epitopes; %d epitopes "
              "qualify at >=%d receptors with >=%d in each receptor-disjoint half, capped at "
              "%d per half -> %d receptors (%d in half 0, %d in half 1)"
              % (rec.height, rec["peptide"].n_unique(), len(pan), MIN_REC, MIN_HALF, CAP // 2,
                 q.height, q.filter(pl.col("half") == 0).height,
                 q.filter(pl.col("half") == 1).height))

        r_int = arm_internal(q, pan, rng)
        r_seen = arm_internal(q, pan, rng, seen=True)
        r_tr, cov, n_tr = arm_transfer(q, pan, imm, "transfer")
        print("training rows %d | IMMREP25 rows whose V and J were both seen in training: %.1f%%"
              % (n_tr, 100 * cov))
        cov_rows.append(dict(chain=chain, gene_coverage=cov, n_train_rows=n_tr))

        a2 = q.filter(pl.col("allele") == "A*02:01")
        arms = [r_int, r_seen, r_tr]
        if a2["peptide"].n_unique() >= PANEL:
            pan2 = panels(a2, np.random.default_rng(SEED))
            imm2 = imm.filter(pl.col("allele") == "A*02:01")
            r_m, cov2, _ = arm_transfer(a2, pan2, imm2, "matched_A0201")
            arms.append(r_m)
        frames += [a.with_columns(chain=pl.lit(chain)) for a in arms if a.height]

    t = pl.concat(frames)
    t.write_csv(os.path.join(RESULTS, "transfer_germline.csv"))

    summ = (t.group_by("chain", "arm")
            .agg(n_epitopes=pl.len(), macro01=pl.col("auc01").mean(),
                 min01=pl.col("auc01").min(), max01=pl.col("auc01").max(),
                 macro_auc=pl.col("auc").mean(),
                 n_at_competitor=(pl.col("auc01") >= COMPETITOR).sum(),
                 n_at_floor=(pl.col("auc01") <= P01_FLOOR + 1e-9).sum())
            .sort("chain", "arm"))
    print("\n=== macro-AUC_0.1 by arm (COMPETITOR=%.2f, metric floor=%.4f) ===" % (COMPETITOR,
                                                                                  P01_FLOOR))
    with pl.Config(tbl_rows=20, tbl_width_chars=220):
        print(summ)
    summ.write_csv(os.path.join(RESULTS, "transfer_germline_summary.csv"))

    # Paired contrast. The two internal arms hold out the SAME epitopes, with the same
    # receptors and the same negative subsets, and differ in exactly one variable: whether the
    # held-out epitope was present in training. Their per-epitope difference is therefore the
    # condition-matched comparison R1.1 asks for, with the epitope as the unit of inference
    # (the choice ImmSET independently recommends, per-peptide AUROC being broadly spread).
    piv = (t.filter(pl.col("arm").is_in(["internal", "internal_seen"]))
           .pivot(on="arm", index=["chain", "peptide"], values="auc01")
           .with_columns(delta=pl.col("internal_seen") - pl.col("internal")))
    assert piv["delta"].null_count() == 0, \
        "the internal arms did not hold out the same epitopes; the pairing is invalid"
    print("\n=== in-distribution minus peptide-unseen, paired by epitope (macro-AUC_0.1) ===")
    pair_macros = {}
    for chain, tag in (("TRB", "B"), ("TRA", "A")):
        s = piv.filter(pl.col("chain") == chain)
        d = s["delta"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        pt = float(ttest_rel(s["internal_seen"].to_numpy(), s["internal"].to_numpy()).pvalue)
        wp = float(wilcoxon(d).pvalue)
        print("  %s, n=%d epitopes: mean %+.4f macro-AUC_0.1 units, 95%% CI [%+.4f, %+.4f], "
              "paired t p=%.2g, Wilcoxon p=%.2g, positive in %d of %d"
              % (chain, len(d), d.mean(), lo, hi, pt, wp, int((d > 0).sum()), len(d)))
        pair_macros.update({
            "xferDelta" + tag: "%.3f" % d.mean(),
            "xferDelta%sLo" % tag: "%.3f" % lo,
            "xferDelta%sHi" % tag: "%.3f" % hi,
            "xferDelta%sP" % tag: "%.1e" % pt,
            "xferDelta%sNpos" % tag: "%d" % int((d > 0).sum()),
            "xferDelta%sNep" % tag: "%d" % len(d),
        })

    def g(chain, arm, col):
        r = summ.filter((pl.col("chain") == chain) & (pl.col("arm") == arm))
        return None if not r.height else r[col][0]

    macros = {}
    for chain, tag in (("TRB", "B"), ("TRA", "A")):
        for arm, atag in (("internal", "Int"), ("internal_seen", "IntS"),
                          ("transfer", "Tr"), ("matched_A0201", "Mat")):
            v = g(chain, arm, "macro01")
            if v is None:
                continue
            macros["xfer%s%s" % (atag, tag)] = "%.3f" % v
            macros["xfer%s%sMax" % (atag, tag)] = "%.3f" % g(chain, arm, "max01")
            macros["xfer%s%sNat" % (atag, tag)] = "%d" % g(chain, arm, "n_at_competitor")
            macros["xfer%s%sNep" % (atag, tag)] = "%d" % g(chain, arm, "n_epitopes")
    macros.update(pair_macros)
    macros["xferMinRec"] = "%d" % MIN_REC
    macros["xferCap"] = "%d" % CAP
    macros["xferCov"] = "%.1f" % (100 * min(c["gene_coverage"] for c in cov_rows))
    with open(os.path.join(ADAT, "transfer_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/transfer_germline.csv, results/transfer_germline_summary.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "transfer_macros.tex"), len(macros)))


def demo():
    """Self-check: gene harmonisation is total, VDJdb does not contain IMMREP25's peptides,
    the mimicked panel construction reproduces IMMREP25's 1:9 positive:negative geometry, and
    a fixed-category design leaves an unseen gene as missing rather than as a silent level."""
    mp = gene_map()
    assert mp.get("TCRBV03-01/03-02") == "TRBV3-1", "ambiguous Adaptive token misresolved"
    imm = immrep_receptors("TRB")
    assert imm.height == 10000 and int(imm["label"].sum()) == 1000, "unexpected benchmark shape"
    assert imm["v"].null_count() == 0 and imm["j"].null_count() == 0, "unmapped genes remain"
    rec = vdjdb_receptors("TRB")
    assert rec.filter(pl.col("peptide").is_in(imm["peptide"].unique().to_list())).height == 0, \
        "VDJdb overlaps IMMREP25's peptides -- the transfer arm would not be blind"
    rng = np.random.default_rng(SEED)
    q = qualifying(rec, rng)
    per = q.group_by("peptide", "half").agg(n=pl.len())
    assert per["n"].max() <= CAP // 2, "per-half cap not applied"
    assert per["n"].min() >= MIN_HALF, "epitope kept with too few receptors in a half"
    # the halves must be disjoint AT RECEPTOR LEVEL, not merely row level
    h0 = set(q.filter(pl.col("half") == 0)["cdr3"].to_list())
    h1 = set(q.filter(pl.col("half") == 1)["cdr3"].to_list())
    assert not (h0 & h1), "a receptor occurs in both halves; the split is not receptor-level"
    # determinism: identical inputs must give identical selections, or the seeds and versions
    # recorded for reproduction are worthless. Both failures this guards against were real --
    # unique() picking an arbitrary duplicate, and a tied sort choosing a different held-out set
    # for each arm.
    q2 = qualifying(rec, np.random.default_rng(SEED))
    assert q.equals(q2), "qualifying() is not reproducible across calls"
    assert panels(q, np.random.default_rng(SEED)) == panels(q2, np.random.default_rng(SEED)), \
        "panels() is not reproducible across calls"
    pan = panels(q, rng)
    ep = sorted(pan)[0]
    te = task_rows([ep], q, pan)
    npos, nneg = int(te["label"].sum()), int((1 - te["label"]).sum())
    assert npos > 0 and nneg > 0, "task rows must carry both classes"
    assert nneg <= NEG * CAP, "more negatives than the panel can supply"
    # geometry matching must land exactly on IMMREP25's 1:NEG whenever the pool allows it
    tem = match_ratio(te, np.random.default_rng(SEED))
    npos_m = int(tem["label"].sum())
    assert npos_m == npos, "ratio matching must not touch the positives"
    assert int((1 - tem["label"]).sum()) == NEG * npos_m, \
        "ratio matching did not reach 1:%d" % NEG
    # an unseen category must arrive as missing, not as a new level
    cats = {"v": ["TRBV1"], "j": ["TRBJ1-1"]}
    probe = te.filter((pl.col("v") != "TRBV1") & (pl.col("j") != "TRBJ1-1")).head(5)
    assert probe.height, "no row available to probe the unseen-category path"
    x, _, _ = _design(probe, cats)
    assert np.isnan(x[:, 0]).all(), "unseen gene leaked in as a category instead of missing"
    print("transfer_germline.demo OK  (%d/%d IMMREP25 gene tokens mapped, 0 VDJdb records on "
          "IMMREP25 peptides, panel task %d positives vs %d negatives for %s)"
          % (imm.height, imm.height, npos, nneg, ep))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
