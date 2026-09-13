#!/usr/bin/env python
# 2026-09-13  Is AlphaFold's confidence on IMMREP25 explained by similarity to the templates it
#             actually retrieved?
#
# ipTM is a confidence in structural correctness, not a binding score, so the natural hypothesis
# is that it is high when the query looks like something already in the PDB. TCRmodel2 records
# the templates it used -- `tcra_tmplts`, `tcrb_tmplts`, `pmhc_tmplts`, four PDB chains each --
# so the hypothesis can be tested against its OWN retrievals rather than against a similarity
# proxy we invent.
#
# Two facts about those retrievals shape the whole analysis, and both are asserted below:
#
#   1. The pMHC template set is a PER-PEPTIDE CONSTANT. All 50 complexes of a peptide share one
#      set (20 peptides -> 20 sets; the B*40:01 peptides share some sets between them). So the
#      pMHC template channel contributes a peptide main effect and CANNOT rank receptors within
#      a peptide -- which is the only variation the benchmark's per-peptide task can use.
#   2. The TCR template sets vary per complex (166 distinct alpha, 183 distinct beta PDBs), and
#      are a function of the receptor. They are therefore the only template channel that can
#      move a within-peptide ranking.
#
# The analysis is consequently run twice: raw, and after centring every variable on its peptide
# mean. The raw correlation is confounded -- between-peptide differences in ipTM are inseparable
# from between-peptide differences in the retrieved template set -- so the WITHIN-PEPTIDE
# correlation is the one that speaks to the benchmark's task. Reporting only the raw number
# would overstate the case.
#
# What this CANNOT do: estimate the receptor x peptide interaction in ipTM. We hold confidence
# for each receptor paired with its own cognate peptide only -- one cell per receptor, the
# diagonal of the receptor x peptide matrix -- so a receptor main effect is not separable from
# an interaction. That needs the cross-pair folds (the sampled 1,000-complex N2 design).
#
# Run: python src/iptm_templates.py      (--demo for the self-check)
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
STATS = os.path.join(REPO, "cache", "struct", "immrep25")
EPIMAP = os.path.join(REPO, "cache", "struct", "immrep25_epitopes.tsv")
CRYSTALS = os.path.expanduser("~/vcs/code/tcren/scratch/markup_2026.csv")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

SLOTS = {"tcra": "cdr3a", "tcrb": "cdr3b", "pmhc": "epitope"}
QUERY = {"tcra": "cdr3a", "tcrb": "cdr3b", "pmhc": "peptide"}


def load_epitope_map() -> dict[str, str]:
    """complex.id -> peptide.

    The file is separated by a LITERAL backslash-t, not a tab -- iptm_compare.py splits on the
    two-character string for the same reason. A naive tab parse yields one column and silently
    maps nothing.
    """
    sep = chr(92) + "t"
    out = {}
    for ln in open(EPIMAP).read().splitlines():
        if sep in ln:
            cid, pep = ln.split(sep)[:2]
            if pep:
                out[cid] = pep
    return out


def load_confidence() -> pl.DataFrame:
    """One row per folded complex: ranked_0 confidences plus the across-model ipTM spread."""
    cid2pep = load_epitope_map()
    rows = []
    for f in sorted(glob.glob(os.path.join(STATS, "*.stats.json"))):
        cid = os.path.basename(f).replace(".stats.json", "")
        d = json.load(open(f))
        r = d.get("ranked_0", {})
        it = [d.get("ranked_%d" % k, {}).get("iptm") for k in range(5)]
        it = [x for x in it if x is not None]
        rows.append(dict(
            cid=cid, peptide=cid2pep.get(cid, ""),
            iptm=r.get("iptm"), tcr_pmhc_iptm=r.get("tcr-pmhc_iptm"),
            plddt=r.get("plddt"), ptm=r.get("ptm"),
            iptm_sd=float(np.std(it, ddof=1)) if len(it) > 1 else None,
            tmpl_tcra=d.get("tcra_tmplts") or [],
            tmpl_tcrb=d.get("tcrb_tmplts") or [],
            tmpl_pmhc=d.get("pmhc_tmplts") or []))
    return pl.DataFrame(rows)


def crystal_lookup() -> dict[str, dict]:
    """pdb id -> {cdr3a, cdr3b, epitope} for the reference complexes that have them."""
    cry = pl.read_csv(CRYSTALS).drop_nulls(["cdr3a", "cdr3b", "peptide"])
    return {r["pdb.id"].lower(): {"cdr3a": r["cdr3a"], "cdr3b": r["cdr3b"],
                                  "epitope": r["peptide"]}
            for r in cry.to_dicts()}


def template_similarity(conf: pl.DataFrame, look: dict[str, dict]) -> pl.DataFrame:
    """Per complex and per slot: the best and mean BLOSUM62 score to the retrieved templates.

    Only templates present in the crystal markup can be scored (a retrieved chain may be an
    unbound TCR or an MHC-only structure, which carries no counterpart sequence). The resolvable
    count is carried through as a column rather than hidden, because it is itself a covariate.
    """
    from retrieval_baseline import _aligner
    al = _aligner()
    bench = (pl.read_csv(BENCH, separator="\t").filter(pl.col("label") == 1)
             .select("tcra_cdr3", "tcrb_cdr3", "peptide", "hla").unique())
    q = {(r["tcra_cdr3"] + "_" + r["tcrb_cdr3"]): r for r in bench.to_dicts()}
    cache: dict[tuple, float] = {}

    def score(a: str, b: str) -> float:
        k = (a, b)
        if k not in cache:
            cache[k] = float(al.score(a, b))
        return cache[k]

    out = []
    for r in conf.to_dicts():
        rec = q.get(r["cid"])
        if rec is None:
            continue
        row = dict(cid=r["cid"], peptide=r["peptide"], hla=rec["hla"],
                   cdr3a=rec["tcra_cdr3"], cdr3b=rec["tcrb_cdr3"],
                   iptm=r["iptm"], tcr_pmhc_iptm=r["tcr_pmhc_iptm"],
                   plddt=r["plddt"], ptm=r["ptm"], iptm_sd=r["iptm_sd"])
        qseq = {"tcra": rec["tcra_cdr3"], "tcrb": rec["tcrb_cdr3"], "pmhc": r["peptide"]}
        for slot, refcol in SLOTS.items():
            vals = []
            for t in r["tmpl_" + slot]:
                ref = look.get(t.split("_")[0].lower())
                if ref is not None:
                    vals.append(score(qseq[slot], ref[refcol]))
            row["%s_best" % slot] = max(vals) if vals else None
            row["%s_mean" % slot] = float(np.mean(vals)) if vals else None
            row["%s_nres" % slot] = len(vals)
        out.append(row)
    return pl.DataFrame(out)


def reference_similarity(t: pl.DataFrame, topk=(1, 3)) -> pl.DataFrame:
    """Top-k mean BLOSUM62 similarity of each query component to the WHOLE crystal reference.

    The retrieved-template features cover only the four chains TCRmodel2 actually used, and just
    8 of the 43 distinct pMHC templates appear in the markup -- so peptide similarity is barely
    resolvable there. Scoring against every reference complex instead measures what was
    AVAILABLE to retrieve, which is what the hypothesis is really about, and it resolves for all
    20 peptides. The reference is retrieval_baseline's own, so these numbers are directly
    comparable with the N1 retrieval baseline rather than being a second, subtly different set.
    """
    from retrieval_baseline import _aligner, _crystal_reference, simmat
    al = _aligner()
    cry = _crystal_reference()
    cols = []
    for slot, refcol in SLOTS.items():
        qcol = QUERY[slot]
        qs = sorted(set(t[qcol].to_list()))
        M = simmat(qs, cry[refcol].tolist(), al)
        srt = -np.sort(-M, axis=1)
        for k in topk:
            d = {q: float(srt[i, :k].mean()) for i, q in enumerate(qs)}
            cols.append(pl.col(qcol).replace_strict(d, default=None)
                        .alias("%s_pdb%d" % (slot, k)))
    print("reference for the availability features: %d crystal complexes" % len(cry))
    return t.with_columns(cols)


def interface_distance(t: pl.DataFrame, topk=(1, 3)) -> pl.DataFrame:
    """Distance from each PREDICTED interface to the nearest real one, in descriptor space.

    The sequence instruments ask whether a query looks like a PDB entry by alignment score. The
    hypothesis is really structural -- ipTM is confidence that the modelled interface is right --
    so the closer instrument is the distance between the predicted complex's own interface
    descriptors and those of the solved complexes. tcren computes the same descriptor family for
    both sets, and they share 50 columns (the predicted table lacks the crystal-only geometry and
    the crystal table lacks the TCRen energy terms), so the space is the shared 50.

    Standardisation uses the CRYSTAL mean and sd: the reference defines the space, and scaling by
    the predicted set's own spread would make the distance depend on the query cohort.
    """
    pred = pl.read_csv(os.path.join(RESULTS, "struct_desc_immrep25.tsv"), separator="\t",
                       infer_schema_length=5000)
    cry = pl.read_csv(os.path.expanduser("~/vcs/code/tcren/reproduce/native_features.tsv"),
                      separator="\t", infer_schema_length=5000)
    num = [c for c in set(pred.columns) & set(cry.columns)
           if c != "complex.id" and pred[c].dtype.is_numeric() and cry[c].dtype.is_numeric()]
    num = sorted(num)
    P = pred.select(num).to_numpy().astype(float)
    C = cry.select(num).to_numpy().astype(float)
    mu = np.nanmean(C, axis=0)
    sd = np.nanstd(C, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    P = np.where(np.isfinite(P), P, mu)          # impute at the reference mean (§11)
    C = np.where(np.isfinite(C), C, mu)
    P = (P - mu) / sd
    C = (C - mu) / sd
    D = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    srt = np.sort(D, axis=1)
    out = {"cid": pred["complex.id"].to_list()}
    for k in topk:
        out["iface_d%d" % k] = srt[:, :k].mean(axis=1).tolist()
    print("interface space: %d shared descriptor columns, %d predicted vs %d reference complexes"
          % (len(num), P.shape[0], C.shape[0]))
    return t.join(pl.DataFrame(out), on="cid", how="left")


def _centre(t: pl.DataFrame, cols) -> pl.DataFrame:
    """Subtract each variable's peptide mean: removes every peptide main effect, including the
    per-peptide constant pMHC template set."""
    return t.with_columns([(pl.col(c) - pl.col(c).mean().over("peptide")).alias(c + "_c")
                           for c in cols])


def eta_sq(t: pl.DataFrame, col: str, group: str = "peptide") -> float:
    """Share of variance in `col` explained by `group` (one-way, descriptive)."""
    y = t[col].to_numpy()
    gm = t.group_by(group).agg(m=pl.col(col).mean(), n=pl.len()).to_dicts()
    tot = float(((y - y.mean()) ** 2).sum())
    between = float(sum(g["n"] * (g["m"] - y.mean()) ** 2 for g in gm))
    return between / tot if tot > 0 else float("nan")


def main():
    conf = load_confidence()
    look = crystal_lookup()
    t = template_similarity(conf, look).drop_nulls(["iptm"])
    t.write_csv(os.path.join(RESULTS, "iptm_templates.csv"))
    print("complexes scored: %d | crystal reference entries: %d" % (t.height, len(look)))
    print("resolvable templates per slot (of 4): " + ", ".join(
        "%s %.2f" % (s, t["%s_nres" % s].mean()) for s in SLOTS))

    # the two structural facts the analysis rests on
    sets_per_pep = (conf.with_columns(s=pl.col("tmpl_pmhc").list.sort().list.join(","))
                    .group_by("peptide").agg(n=pl.col("s").n_unique()))
    assert sets_per_pep["n"].max() == 1, "pMHC template set is not constant within a peptide"
    print("pMHC template sets per peptide: max %d (constant) | distinct sets overall: %d"
          % (sets_per_pep["n"].max(),
             conf.with_columns(s=pl.col("tmpl_pmhc").list.sort().list.join(","))["s"].n_unique()))

    eta = eta_sq(t, "iptm")
    print("\nvariance of ipTM explained by peptide identity: eta^2 = %.3f" % eta)

    t = interface_distance(reference_similarity(t))
    feats = [f for f in ["tcra_best", "tcrb_best", "pmhc_best",
                         "tcra_pdb1", "tcrb_pdb1", "pmhc_pdb1",
                         "tcra_pdb3", "tcrb_pdb3", "pmhc_pdb3",
                         "iface_d1", "iface_d3"]
             if t[f].null_count() < t.height]
    # Per-feature complete cases, NOT a global drop_nulls: only the pMHC template features are
    # sparse (0.8 of 4 templates resolve), and dropping those rows would have cost the TCR and
    # interface arms 45% of the sample for a covariate they never use (§11).
    tc = _centre(t, feats + ["iptm", "tcr_pmhc_iptm"])
    rows = []
    for f in feats:
        for target in ["iptm", "tcr_pmhc_iptm"]:
            sub = tc.drop_nulls([f, target])
            x, y = sub[f].to_numpy(), sub[target].to_numpy()
            xc, yc = sub[f + "_c"].to_numpy(), sub[target + "_c"].to_numpy()
            # a feature that is constant within peptide (any peptide-level quantity) has no
            # within-peptide variation at all; report that as missing rather than as NaN
            within = xc.std() > 1e-12 and yc.std() > 1e-12
            rows.append(dict(feature=f, target=target, n=len(x),
                             r_raw=float(pearsonr(x, y)[0]), p_raw=float(pearsonr(x, y)[1]),
                             rho_raw=float(spearmanr(x, y)[0]),
                             r_within=float(pearsonr(xc, yc)[0]) if within else None,
                             p_within=float(pearsonr(xc, yc)[1]) if within else None))
    c = pl.DataFrame(rows)
    c.write_csv(os.path.join(RESULTS, "iptm_templates_corr.csv"))
    print("\n=== BLOSUM similarity to the RETRIEVED templates vs confidence ===")
    print("r_raw is confounded by peptide (each peptide has its own fixed pMHC template set);")
    print("r_within is computed after centring every variable on its peptide mean.")
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=4):
        print(c)

    # Per-peptide test: peptide identity carries ~eta^2 of the ipTM variance, and the pMHC
    # template set is a per-peptide constant, so the hypothesis "more like the PDB, more
    # confident" is properly tested at the peptide level with the peptide as the unit.
    pep = (t.group_by("peptide", "hla")
           .agg(n=pl.len(), iptm_med=pl.col("iptm").median(),
                pep_pdb1=pl.col("pmhc_pdb1").first(), pep_pdb3=pl.col("pmhc_pdb3").first())
           .sort("iptm_med", descending=True))
    print("\n=== per-peptide: best BLOSUM to the PDB reference peptides vs median ipTM ===")
    with pl.Config(tbl_rows=25, tbl_width_chars=200, float_precision=4):
        print(pep)
    rp = pearsonr(pep["pep_pdb1"].to_numpy(), pep["iptm_med"].to_numpy())
    rs = spearmanr(pep["pep_pdb1"].to_numpy(), pep["iptm_med"].to_numpy())
    print("n=%d peptides: Pearson r=%.3f (p=%.3g), Spearman rho=%.3f (p=%.3g)"
          % (pep.height, rp[0], rp[1], rs[0], rs[1]))

    def pick(f, target, col):
        """NaN for both "no such row" and "structurally undefined" -- a peptide-level feature
        has no within-peptide variation at all, so its r_within is None by construction."""
        r = c.filter((pl.col("feature") == f) & (pl.col("target") == target))
        if not r.height or r[col][0] is None:
            return float("nan")
        return float(r[col][0])

    macros = {
        "itplN": "%d" % t.height,
        "itplEtaPep": "%.3f" % eta,
        "itplPmhcSets": "%d" % conf.with_columns(
            s=pl.col("tmpl_pmhc").list.sort().list.join(","))["s"].n_unique(),
        "itplTcraRaw": "%.3f" % pick("tcra_best", "iptm", "r_raw"),
        "itplTcraWithin": "%.3f" % pick("tcra_best", "iptm", "r_within"),
        "itplTcrbRaw": "%.3f" % pick("tcrb_best", "iptm", "r_raw"),
        "itplTcrbWithin": "%.3f" % pick("tcrb_best", "iptm", "r_within"),
        "itplPepRaw": "%.3f" % pick("pmhc_best", "iptm", "r_raw"),
        "itplPepN": "%d" % int(pick("pmhc_best", "iptm", "n")),
        "itplIptmMean": "%.3f" % float(t["iptm"].mean()),
        "itplIptmSd": "%.3f" % float(t["iptm"].std()),
        "itplPepPdbR": "%.3f" % float(rp[0]),
        "itplPepPdbP": "%.3g" % float(rp[1]),
        "itplPepPdbRho": "%.3f" % float(rs[0]),
        "itplNpep": "%d" % pep.height,
        "itplTcraAvailWithin": "%.3f" % pick("tcra_pdb1", "iptm", "r_within"),
        "itplTcrbAvailWithin": "%.3f" % pick("tcrb_pdb1", "iptm", "r_within"),
        "itplIfaceRaw": "%.3f" % pick("iface_d1", "iptm", "r_raw"),
        "itplIfaceWithin": "%.3f" % pick("iface_d1", "iptm", "r_within"),
        "itplIfaceP": "%.3g" % pick("iface_d1", "iptm", "p_raw"),
        "itplIfaceN": "%d" % int(pick("iface_d1", "iptm", "n")),
    }
    with open(os.path.join(ADAT, "iptm_templates_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/iptm_templates.csv, results/iptm_templates_corr.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "iptm_templates_macros.tex"), len(macros)))


def demo():
    """Self-check: the epitope map parses despite its literal backslash-t, confidences load for
    all 1,000 folded complexes, the pMHC template set is a per-peptide constant, and peptide
    centring removes the peptide main effect exactly (eta^2 -> 0)."""
    m = load_epitope_map()
    assert len(m) >= 1000, "epitope map parsed only %d rows -- check the separator" % len(m)
    conf = load_confidence()
    assert conf.height == 1000, "expected 1000 folded complexes, got %d" % conf.height
    assert conf["iptm"].null_count() == 0, "some complexes carry no ranked_0 ipTM"
    assert conf["peptide"].n_unique() == 20, "expected 20 peptides among the folded complexes"
    per = (conf.with_columns(s=pl.col("tmpl_pmhc").list.sort().list.join(","))
           .group_by("peptide").agg(n=pl.col("s").n_unique()))
    assert per["n"].max() == 1, "pMHC template set varies within a peptide"
    assert conf["tmpl_tcra"].list.len().min() == 4, "expected four alpha templates per complex"
    # centring must annihilate the peptide main effect
    tc = _centre(conf.select("peptide", "iptm"), ["iptm"])
    assert eta_sq(tc, "iptm") > 0.01, "peptide should explain some raw ipTM variance"
    assert eta_sq(tc, "iptm_c") < 1e-12, \
        "peptide centring did not remove the peptide main effect"
    look = crystal_lookup()
    assert len(look) > 300, "crystal lookup is too small (%d)" % len(look)
    print("iptm_templates.demo OK  (%d complexes, %d peptides, pMHC template set constant "
          "within every peptide, %d crystal reference entries, peptide centring exact)"
          % (conf.height, conf["peptide"].n_unique(), len(look)))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
