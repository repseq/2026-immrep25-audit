#!/usr/bin/env python
# 2026-09-14  Is a structural confidence score transferable across epitopes? The decomposition,
# for interface ipTM and for global pLDDT.
#
# The benchmark reports a PER-EPITOPE AUC, and a per-epitope AUC is rank-based within that
# epitope: it is invariant to any per-epitope shift of the score. So a method can post a good
# per-epitope number while having no operating point that works across epitopes at all. The
# obvious way to test that -- take the AUC over all records pooled, and subtract -- is wrong, and
# this module exists to replace it with the exact decomposition.
#
# The pooled AUC is a Mann-Whitney statistic, so it decomposes over ORDERED epitope pairs:
#
#     AUC_pooled = sum_e sum_f w_ef * U_ef,    w_ef = (n_pos_e * n_neg_f) / (N_pos * N_neg)
#
# with U_ef = P(binder drawn from e outranks non-binder drawn from f). The diagonal terms are the
# within-epitope AUCs; the off-diagonal terms are the cross-epitope comparisons. Hence
#
#     AUC_pooled = W_diag * within_weighted + (1 - W_diag) * cross,   W_diag = sum_e w_ee
#
# and the weights are DETERMINED, not chosen. That matters because with E roughly balanced
# cohorts W_diag ~ 1/E, so a pooled AUC over 26 cohorts is ~96% cross-epitope comparisons. A
# "within minus pooled" difference therefore mixes three things at once: the between-epitope
# offset penalty, a reweighting from macro to pair-weighted, and a 96/4 blend of two different
# estimands. A near-zero difference does not mean the score transfers; it means those cancelled.
#
# The decomposition computes all of it -- within_macro, within_weighted, cross, W_diag -- and the
# identity is exact, so it doubles as the self-check: the reconstruction must equal sklearn's
# pooled AUC to floating point.
#
# WHAT IS REPORTED, however, is the SPREAD alone. Measured here, the naive difference is NEGATIVE
# for both scores: the pooled AUC exceeds the macro within-epitope mean, because at ~6% diagonal
# weight the pooled figure is almost entirely cross-epitope while the macro mean is dragged down
# by small cohorts the pair weighting barely counts. So the difference does not measure an offset
# penalty and no gap macro is emitted -- the decomposition stays here and in the CSV as the reason
# for that, not as something the text can quote.
#
# The reportable result is that a per-epitope AUC from one scorer ranges over most of the unit
# interval across cohorts. That is what defeats ranking, and it needs no decomposition to state.
#
# ipTM is the primary score. Our pLDDT is a global model-quality figure, not restricted to the
# CDR3-peptide interface, so it is the weaker instrument for a binding question and is reported
# alongside rather than as the headline.
#
# Run: python src/transferability.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
HF = os.path.expanduser("~/hf/tcren_structures")
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
# paths.TABLES honours AUDIT_MS_REPO, so a worktree session writes the table where it can
# see it. The other emitters in this repo still hardcode the MAIN checkout -- deliberately
# not fixed here, but it is a live trap: regenerating their tables from a worktree is
# invisible to the worktree build.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
POS = os.path.join(RESULTS, "iptm_templates.csv")

MIN_CLASS = 15           # per-epitope minimum per class, as in plddt_panels/iptm_compare
SCORES = ("iptm", "plddt")
SETS = [
    ("TCRvdb (assay-labelled)", "tcrvdb/metadata.tsv", "y"),
    ("VDJdb benchmark (mock negatives)", "vdjdb_binder_benchmark/metadata.tsv", "y"),
    ("VDJdb free pool (mispairings)", "vdjdb_free_pool/metadata.tsv", "y"),
]


def load() -> pd.DataFrame:
    """Every labelled reference structure carrying both scores, plus IMMREP25's positives."""
    frames = []
    for label, path, ycol in SETS:
        p = os.path.join(HF, path)
        t = pd.read_csv(p, sep="\t", low_memory=False)
        missing = [c for c in SCORES if c not in t.columns]
        assert not missing, "%s lacks %s" % (label, missing)
        t = (t.rename(columns={ycol: "y"})[["y", "epitope"] + list(SCORES)]
             .dropna(subset=["y", "epitope"] + list(SCORES))
             .assign(source=label))
        frames.append(t)
    imm = pd.read_csv(POS)
    frames.append(imm.rename(columns={"peptide": "epitope"})[["epitope"] + list(SCORES)]
                  .assign(y=1, source="IMMREP25"))
    d = pd.concat(frames, ignore_index=True)
    d["y"] = d.y.astype(int)
    return d


def paired(d: pd.DataFrame) -> pd.DataFrame:
    """Cohorts with both classes at >= MIN_CLASS: the ones a within-epitope AUC exists for."""
    g = (d.groupby(["source", "epitope"]).y.agg(n_pos="sum", n="size").reset_index())
    g["n_neg"] = g.n - g.n_pos
    return g[(g.n_pos >= MIN_CLASS) & (g.n_neg >= MIN_CLASS)].copy()


def u_stat(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y), the Mann-Whitney functional, by ranking once."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    both = np.concatenate([x, y])
    r = pd.Series(both).rank(method="average").to_numpy()
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def decompose(d: pd.DataFrame, cells: pd.DataFrame, score: str) -> dict:
    """The exact epitope-pair decomposition of the pooled AUC."""
    keys = list(zip(cells.source, cells.epitope))
    pos = {k: d[(d.source == k[0]) & (d.epitope == k[1]) & (d.y == 1)][score].to_numpy()
           for k in keys}
    neg = {k: d[(d.source == k[0]) & (d.epitope == k[1]) & (d.y == 0)][score].to_numpy()
           for k in keys}
    npos = np.array([len(pos[k]) for k in keys], dtype=float)
    nneg = np.array([len(neg[k]) for k in keys], dtype=float)
    W = np.outer(npos, nneg) / (npos.sum() * nneg.sum())
    U = np.array([[u_stat(pos[a], neg[b]) for b in keys] for a in keys])

    diag = np.diag_indices(len(keys))
    w_diag = float(W[diag].sum())
    within_w = float((W[diag] * U[diag]).sum() / w_diag)
    off = W.copy(); off[diag] = 0.0
    cross = float((off * U).sum() / off.sum())
    pooled_recon = float((W * U).sum())

    sub = d[d.set_index(["source", "epitope"]).index.isin(keys)]
    pooled_sk = float(roc_auc_score(sub.y, sub[score]))
    return dict(score=score, n_cohorts=len(keys), n_records=int(len(sub)),
                within_macro=float(np.mean(U[diag])),
                within_sd=float(np.std(U[diag], ddof=1)),
                within_lo=float(np.min(U[diag])), within_hi=float(np.max(U[diag])),
                within_weighted=within_w, cross=cross, w_diag=w_diag,
                pooled=pooled_sk, pooled_reconstructed=pooled_recon,
                identity_err=abs(pooled_recon - pooled_sk),
                gap_naive=float(np.mean(U[diag])) - pooled_sk, U=U, keys=keys, diag=diag)


def main():
    d = load()
    cells = paired(d)
    print("=== cohorts with both classes at >= %d (a within-epitope AUC exists) ===" % MIN_CLASS)
    print(cells.groupby("source").agg(cohorts=("epitope", "size"),
                                      pos=("n_pos", "sum"), neg=("n_neg", "sum")).to_string())

    res, per_ep = [], []
    for s in SCORES:
        r = decompose(d, cells, s)
        U, diag, keys = r.pop("U"), r.pop("diag"), r.pop("keys")
        res.append(r)
        for (src, ep), v in zip(keys, U[diag]):
            per_ep.append(dict(score=s, source=src, epitope=ep, within_auc=v))
        print("\n=== %s ===" % s)
        print("  within-epitope AUC, macro over %d cohorts: %.4f (sd %.4f, range %.4f-%.4f)"
              % (r["n_cohorts"], r["within_macro"], r["within_sd"], r["within_lo"], r["within_hi"]))
        print("  the same diagonal as it enters the pooled statistic: %.4f" % r["within_weighted"])
        print("  CROSS-epitope (transferability proper):            %.4f" % r["cross"])
        print("  pooled AUC %.4f  = %.4f*%.4f + %.4f*%.4f   [identity err %.2e]"
              % (r["pooled"], r["w_diag"], r["within_weighted"], 1 - r["w_diag"], r["cross"],
                 r["identity_err"]))
        print("  only %.1f%% of the pooled statistic is within-epitope at all, so the naive "
              "within-minus-pooled difference (%+.4f) is not an offset penalty"
              % (100 * r["w_diag"], r["gap_naive"]))

    t = pd.DataFrame(res)
    t.to_csv(os.path.join(RESULTS, "transferability.csv"), index=False)
    pe = pd.DataFrame(per_ep)
    pe.to_csv(os.path.join(RESULTS, "transferability_per_epitope.csv"), index=False)

    print("\n=== within-epitope AUC spread per cohort, both scores ===")
    spread = (pe.groupby(["source", "score"]).within_auc
              .agg(n="size", lo="min", median="median", hi="max",
                   sd=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)
              .reset_index().pivot(index="source", columns="score"))
    print(spread.to_string())

    print("\n=== where each set's scores sit, as raw values (is the SCALE shared?) ===")
    for s in SCORES:
        line = []
        for src, g in d.groupby("source"):
            b = g[g.y == 1][s]
            line.append("%s %.3f" % (src.split(" (")[0], float(b.median())))
        print("  %-6s binder medians: %s" % (s, " | ".join(line)))

    # SPREAD ONLY. The decomposition's cross/pooled/W_diag terms stay in the CSV; emitting them
    # would let the text quote a "gap" that this module's own numbers show to be uninformative.
    #
    # 2026-09-14: three groups ADDED, none of them that gap. (a) trf*Mean / trfN*Rec -- per-arm
    # MEAN within-epitope AUC and record count. The committed trf*Med are MEDIANS, equal to the
    # mean only for the 2-cohort TCRvdb arm (0.7953); the 20-cohort mock arm is median 0.5379
    # against mean 0.5841. Prose must not mix the two, so the mean is emitted explicitly to
    # pair with the SD already emitted. (b) trfSep* -- median binder-minus-nonbinder confidence
    # per arm: a distribution statistic, not a term of the AUC decomposition. (c) trfFit* --
    # the ipTM~pLDDT fit, a relation BETWEEN the two scores, which cannot express a
    # within-vs-pooled gap in either. The bar stands: nothing below emits cross, pooled or
    # W_diag.
    TT_SRC = "TCRvdb (assay-labelled)"
    BM_SRC = "VDJdb benchmark (mock negatives)"
    FP_SRC = "VDJdb free pool (mispairings)"
    TAG = {"TCRvdb (assay-labelled)": "Tt",
           "VDJdb benchmark (mock negatives)": "Bm",
           "VDJdb free pool (mispairings)": "Fp"}
    SC = {"iptm": "Iptm", "plddt": "Pld"}
    macros = {"trfNcohort": "%d" % int(t[t.score == "iptm"].n_cohorts.iloc[0]),
              "trfMinClass": "%d" % MIN_CLASS}
    for s, stag in SC.items():
        r = t[t.score == s].iloc[0]
        macros["trf%sWithin" % stag] = "%.4f" % float(r.within_macro)
        macros["trf%sWithinSd" % stag] = "%.4f" % float(r.within_sd)
        macros["trf%sWithinLo" % stag] = "%.4f" % float(r.within_lo)
        macros["trf%sWithinHi" % stag] = "%.4f" % float(r.within_hi)
    for src, tag in TAG.items():
        g = pe[pe.source == src]
        macros["trfN%s" % tag] = "%d" % int(g[g.score == "iptm"].shape[0])
        for s, stag in SC.items():
            v = g[g.score == s].within_auc.to_numpy()
            macros["trf%s%sLo" % (stag, tag)] = "%.4f" % float(v.min())
            macros["trf%s%sHi" % (stag, tag)] = "%.4f" % float(v.max())
            macros["trf%s%sMed" % (stag, tag)] = "%.4f" % float(np.median(v))
            macros["trf%s%sSd" % (stag, tag)] = (
                "%.4f" % float(np.std(v, ddof=1)) if len(v) > 1 else "n/a")
    # (a) per-arm MEAN within-epitope AUC and counts (the medians are emitted above)
    ARM = {"Tt": [TT_SRC], "Bm": [BM_SRC], "Fp": [FP_SRC], "Cb": [BM_SRC, FP_SRC]}
    for tag, srcs in ARM.items():
        cc = cells[cells.source.isin(srcs)]
        macros["trfN%sCoh" % tag] = "%d" % int(len(cc))
        for s2, stag in SC.items():
            rr = decompose(d, cc, s2)
            macros["trf%s%sMean" % (stag, tag)] = "%.4f" % rr["within_macro"]
            if tag == "Cb":
                macros["trf%sCbSd" % stag] = "%.4f" % rr["within_sd"]
                macros["trf%sCbLo" % stag] = "%.4f" % rr["within_lo"]
                macros["trf%sCbHi" % stag] = "%.4f" % rr["within_hi"]
            if s2 == "iptm":
                macros["trfN%sRec" % tag] = "%d" % int(rr["n_records"])

    # (b) median binder-minus-nonbinder separation per arm. IMMREP25 has no negatives in any
    # structure set we hold, so it gets a level and no separation -- that asymmetry is the point.
    for tag, srcs in (("Tt", [TT_SRC]), ("Bm", [BM_SRC]), ("Fp", [FP_SRC])):
        g2 = d[d.source.isin(srcs)]
        pp, nn = g2[g2.y == 1], g2[g2.y == 0]
        macros["trfSepIptm%s" % tag] = "%.4f" % (pp.iptm.median() - nn.iptm.median())
        macros["trfSepPld%s" % tag] = "%.4f" % (pp.plddt.median() - nn.plddt.median())
        macros["trfNPos%s" % tag] = "%d" % len(pp)
        macros["trfNNeg%s" % tag] = "%d" % len(nn)
    gi = d[d.source == "IMMREP25"]
    macros["trfImmIptmMed"] = "%.4f" % gi.iptm.median()
    macros["trfImmPldMed"] = "%.4f" % gi.plddt.median()
    macros["trfNImm"] = "%d" % len(gi)

    # (c) ipTM ~ pLDDT. The IMMREP25 winner scored with a pLDDT statistic, so whether the two
    # confidence channels co-vary decides whether our ipTM analysis speaks to that entry at all.
    FITTAG = {TT_SRC: "Tt", BM_SRC: "Bm", FP_SRC: "Fp", "IMMREP25": "Imm"}
    fitrows = []
    for src, g2 in list(d.groupby("source")) + [("POOLED", d)]:
        x = g2.plddt.to_numpy(float)
        y = g2.iptm.to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        r_ = float(stats.pearsonr(x, y)[0])
        rho_ = float(stats.spearmanr(x, y)[0])
        b_, a_ = (float(v) for v in np.polyfit(x, y, 1))
        tag = "Pool" if src == "POOLED" else FITTAG[src]
        macros["trfFitR%s" % tag] = "%.3f" % r_
        macros["trfFitRho%s" % tag] = "%.3f" % rho_
        macros["trfFitSlope%s" % tag] = "%.4f" % b_
        macros["trfFitRsq%s" % tag] = "%.3f" % (r_ * r_)
        macros["trfFitN%s" % tag] = "%d" % len(x)
        fitrows.append((src, len(x), r_, rho_, b_, r_ * r_))
    rs = [f[2] for f in fitrows[:-1]]
    sl = [f[4] for f in fitrows[:-1]]
    macros["trfFitRLo"] = "%.3f" % min(rs)
    macros["trfFitRHi"] = "%.3f" % max(rs)
    macros["trfFitSlopeLo"] = "%.4f" % min(sl)
    macros["trfFitSlopeHi"] = "%.4f" % max(sl)
    macros["trfFitNarm"] = "%d" % len(rs)

    print("\n=== ipTM ~ pLDDT, added 2026-09-14 ===")
    for src, n_, r_, rho_, b_, r2_ in fitrows:
        print("  %-34s n=%5d r=%.3f rho=%.3f slope=%.4f r2=%.3f"
              % (str(src)[:34], n_, r_, rho_, b_, r2_))

    # the three-arm table, written where the CALLER can see it (honours AUDIT_MS_REPO).
    # Raw strings throughout: an earlier version assembled these by substitution and Python
    # turned \b and \t into BACKSPACE and TAB before the file was written.
    os.makedirs(paths.TABLES, exist_ok=True)
    _tbl = os.path.join(paths.TABLES, "confidence_transfer_table.tex")
    with open(_tbl, "w") as fh:
        fh.write("%% GENERATED by src/transferability.py -- do not edit.\n"
                 "%% Regenerate: python src/transferability.py (2026-immrep25-audit repo).\n")
        fh.write(r"\begin{tabular}{lrrrrr}" + "\n")
        fh.write(r"\toprule" + "\n")
        fh.write(r"negative class & cohorts & records & within-epitope "
                 r"macro-AUC & SD & median $\Delta$ \\" + "\n")
        fh.write(r"\midrule" + "\n")
        for lab, tag in (("biological (invalidated real pairs)", "Tt"),
                         ("combinatorial (shuffled receptors)", "Bm"),
                         ("combinatorial (mispaired chains)", "Fp")):
            fh.write("%s & %s & %s & %s & %s & %s %s\n"
                     % (lab, macros["trfN%sCoh" % tag], macros["trfN%sRec" % tag],
                        macros["trfIptm%sMean" % tag], macros["trfIptm%sSd" % tag],
                        macros["trfSepIptm%s" % tag], r"\\"))
        fh.write(r"\midrule" + "\n")
        fh.write("none exist (IMMREP25) & -- & %s & -- & -- & -- %s\n"
                 % (macros["trfNImm"], r"\\"))
        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")
    print("wrote %s" % _tbl)

    with open(os.path.join(ADAT, "transferability_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/transferability.csv, results/transferability_per_epitope.csv")
    print("wrote %s (%d macros)"
          % (os.path.join(ADAT, "transferability_macros.tex"), len(macros)))


def demo():
    """Self-check. The load-bearing assertion is the identity: the weighted reconstruction over
    epitope pairs must equal sklearn's pooled AUC to floating point. If it does not, the weights
    are wrong and every number this module reports is wrong with them."""
    rng = np.random.default_rng(0)
    # a synthetic two-epitope case with a deliberate per-epitope OFFSET: within-epitope AUC is
    # perfect in both, but no single threshold separates them
    a_pos, a_neg = rng.normal(10, 0.1, 40), rng.normal(9, 0.1, 60)
    b_pos, b_neg = rng.normal(5, 0.1, 30), rng.normal(4, 0.1, 50)
    d = pd.DataFrame(dict(
        source="S", epitope=["A"] * 100 + ["B"] * 80,
        y=[1] * 40 + [0] * 60 + [1] * 30 + [0] * 50,
        iptm=np.concatenate([a_pos, a_neg, b_pos, b_neg]),
        plddt=np.concatenate([a_pos, a_neg, b_pos, b_neg])))
    cells = paired(d)
    assert len(cells) == 2, cells
    r = decompose(d, cells, "iptm")
    # 1. THE identity
    assert r["identity_err"] < 1e-12, "decomposition does not reproduce the pooled AUC (%.3e)" % (
        r["identity_err"])
    # 2. the diagonal is perfect by construction, and the cross term is what the offset destroys
    assert r["within_macro"] > 0.999, "within-epitope AUC should be ~1 here (%.4f)" % (
        r["within_macro"])
    assert r["cross"] < 0.6, "an offset this large must wreck the cross term (%.4f)" % r["cross"]
    assert r["pooled"] < r["within_macro"], "the pooled AUC cannot exceed a perfect diagonal"
    # 3. with NO offset the cross term must recover the diagonal
    d2 = d.copy()
    d2.loc[d2.epitope == "B", "iptm"] += 5.0
    r2 = decompose(d2, cells, "iptm")
    assert r2["identity_err"] < 1e-12, "identity fails on the aligned case"
    assert abs(r2["cross"] - r2["within_macro"]) < 0.02, (
        "aligned scales should make cross ~ within (%.4f vs %.4f)" % (r2["cross"],
                                                                     r2["within_macro"]))
    # 4. u_stat agrees with sklearn, ties included
    x, y = rng.integers(0, 3, 50).astype(float), rng.integers(0, 3, 70).astype(float)
    mine = u_stat(x, y)
    theirs = roc_auc_score(np.r_[np.ones(50), np.zeros(70)], np.r_[x, y])
    assert abs(mine - theirs) < 1e-12, "u_stat disagrees with sklearn on ties (%.9f vs %.9f)" % (
        mine, theirs)
    # 5. W_diag is a probability and the two blocks sum to 1
    assert 0 < r["w_diag"] < 1, r["w_diag"]
    print("transferability.demo OK  (pair decomposition reproduces sklearn's pooled AUC to "
          "%.1e; a per-epitope offset leaves within %.4f but drops cross to %.4f, and removing "
          "it restores cross to %.4f; u_stat matches sklearn with ties)"
          % (r["identity_err"], r["within_macro"], r["cross"], r2["cross"]))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
