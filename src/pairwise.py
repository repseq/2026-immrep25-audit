#!/usr/bin/env python
# 2026-09-13  The benchmark's own task, on the features a recognition function would use.
#
# Every probe in the audit so far asks whether an epitope's receptors resemble one another.
# That is not what the benchmark asks. Its task is PAIRWISE: given a receptor and a peptide,
# do they bind? A reviewer's strongest methodological point is that within-epitope clustering
# and pairwise discrimination are different questions, and that cognate receptors might be
# recognisable through nonlinear interactions among CDR1, CDR2, the peptide and the HLA even
# when no within-epitope convergence is detectable. This module tests that claim directly, on
# the benchmark's own 10,000 labelled pairs, with a learner able to represent such
# interactions.
#
# No annotation work is needed: the released file already carries tcra_cdr1/cdr2,
# tcrb_cdr1/cdr2, the full trimmed chain sequences, hla and hla_sequence alongside the CDR3s.
# So the feature families the objection names are directly available, and the answer is a
# measurement rather than an argument.
#
# Three settings, and the difference between them is the whole point:
#
#   cv_grouped   grouped 5-fold, all peptides seen -- the IN-DISTRIBUTION ceiling.
#   lopo         leave-one-peptide-out: the peptide is unseen, which is the challenge's
#                condition. Its receptors still occur in training as negatives of the other
#                nine same-MHC peptides, because that is how the benchmark is built.
#   lopo_strict  as lopo, but the held-out peptide's 50 cognate receptors are removed from
#                training entirely, so neither the peptide nor its binders are ever seen.
#
# Grouping is BY RECEPTOR, not by row -- but not for the reason one would expect, and the
# self-check measures which reason holds rather than asserting it.
#
# The expected reason is leakage: a receptor appears in ten rows, so an ungrouped split puts
# the same receptor on both sides, and one anticipates a model scoring by receptor identity
# without reading a peptide (the mode ImmSET's appendix warns of -- Garcia Noceda et al. 2026,
# arXiv:2603.26994, Appendix B.3). Measured here, that is NOT what happens. At matched fold
# sizes the ungrouped split scores LOWER (0.596) than the grouped one (0.674), and the pure
# receptor-identity channel -- scoring a test receptor by how often it was positive in
# training -- sits at the McClish floor of 0.474 with a full AUC of 0.40-0.44, i.e. below
# chance.
#
# The explanation is this benchmark's own geometry, and it is the row-level mirror of the rank
# identity in epitope_free.py. Every receptor is positive exactly once and negative exactly
# nine times, so its training positive-rate is ~0.1 for EVERY receptor: receptor identity
# carries no information by construction. What an ungrouped split does instead is put
# contradictory labels on near-identical feature vectors -- the same receptor as a positive for
# its own peptide and a negative for nine others, split across train and test -- which
# degrades the fit rather than inflating it. Grouping is therefore still correct, because it
# avoids that contradiction; it simply is not correcting an inflation.
#
# Run: python src/pairwise.py      (--demo for the self-check)
from __future__ import annotations
import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
PROP_TABLE = os.path.expanduser(
    "~/vcs/code/vdjtools/python/vdjtools/resources/aa_property_table.txt")

MAX_FPR = 0.1
AUC_BEST = 0.60      # best of 126 IMMREP25 submissions (Richardson 2026)
GERM_CV = 0.6441     # the germline-only in-distribution value, for reference
N_FOLDS = 5
SEED = 0
AA = list("ACDEFGHIKLMNPQRSTVWY")
# the loop-bearing regions the objection names, plus the peptide
SEGMENTS = ("tcra_cdr1", "tcra_cdr2", "tcra_cdr3", "tcrb_cdr1", "tcrb_cdr2", "tcrb_cdr3",
            "peptide")
PROPS = ("hydropathy", "charge", "polarity", "volume", "strength")


def _props() -> pd.DataFrame:
    raw = open(PROP_TABLE, newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("##")]
    t = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t").set_index("amino_acid")
    return t[list(PROPS)].astype(float)


def features(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Composition, length and physicochemistry of every segment, plus germline and HLA.

    A gradient-boosted tree forms its own interactions, so the two sides of the interface are
    supplied as separate features rather than pre-multiplied: that is what lets the model use
    a CDR1-by-peptide or HLA-by-V-gene dependence if one exists.
    """
    P = _props()
    lut = {a: P.loc[a].to_numpy() for a in P.index}
    cols = {}
    for seg in SEGMENTS:
        s = d[seg].astype(str)
        cols[f"{seg}_len"] = s.str.len().to_numpy(float)
        comp = np.zeros((len(s), len(AA)), dtype=np.float32)
        phys = np.zeros((len(s), len(PROPS)), dtype=np.float32)
        for i, seq in enumerate(s):
            n = max(len(seq), 1)
            for c in seq:
                if c in lut:
                    phys[i] += lut[c]
                j = AA.index(c) if c in AA else -1
                if j >= 0:
                    comp[i, j] += 1.0
            comp[i] /= n
            phys[i] /= n
        for j, a in enumerate(AA):
            cols[f"{seg}_f{a}"] = comp[:, j]
        for j, p in enumerate(PROPS):
            cols[f"{seg}_{p}"] = phys[:, j]
    X = pd.DataFrame(cols, index=d.index)
    cat = ["tcra_v", "tcra_j", "tcrb_v", "tcrb_j", "hla"]
    for c in cat:
        X[c] = d[c].astype("category")
    return X, cat


def _model(cat):
    return HistGradientBoostingClassifier(categorical_features=cat, max_iter=300,
                                          learning_rate=0.05, random_state=SEED)


def _macro(d: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    rows = []
    for p in sorted(d.peptide.unique()):
        m = (d.peptide == p).to_numpy()
        y = d.label.to_numpy()[m]
        if y.min() == y.max():
            continue
        rows.append(dict(peptide=p, hla=d.hla[m].iloc[0],
                         auc01=roc_auc_score(y, score[m], max_fpr=MAX_FPR),
                         auc=roc_auc_score(y, score[m])))
    return pd.DataFrame(rows)


def run_cv(d, X, cat, groups) -> pd.DataFrame:
    """Grouped K-fold with every peptide seen: the in-distribution ceiling."""
    score = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, d.label, groups):
        m = _model(cat).fit(X.iloc[tr], d.label.iloc[tr])
        score[te] = m.predict_proba(X.iloc[te])[:, 1]
    return _macro(d, score)


def run_lopo(d, X, cat, strict: bool) -> pd.DataFrame:
    """Leave-one-peptide-out. With strict=True the held-out peptide's cognate receptors are
    removed from training too, so neither the peptide nor its binders are ever seen."""
    rec = list(zip(d.tcra_cdr3, d.tcrb_cdr3))
    score = np.zeros(len(d))
    for p in sorted(d.peptide.unique()):
        te = (d.peptide == p).to_numpy()
        tr = ~te
        if strict:
            cognate = {r for r, m, y in zip(rec, te, d.label) if m and y == 1}
            tr = tr & np.array([r not in cognate for r in rec])
        m = _model(cat).fit(X[tr], d.label[tr])
        score[te] = m.predict_proba(X[te])[:, 1]
    return _macro(d, score)


def main():
    d = pd.read_csv(BENCH, sep="\t")
    assert len(d) == 10000 and d.label.sum() == 1000, (len(d), d.label.sum())
    X, cat = features(d)
    assert not X[[c for c in X.columns if c not in cat]].isna().any().any()
    groups = pd.Series(list(zip(d.tcra_cdr3, d.tcrb_cdr3)), index=d.index).astype(str)
    print("pairs %d | features %d (%d categorical) | receptors %d"
          % (len(d), X.shape[1], len(cat), groups.nunique()))

    settings = {
        "cv_grouped": run_cv(d, X, cat, groups),
        "lopo": run_lopo(d, X, cat, strict=False),
        "lopo_strict": run_lopo(d, X, cat, strict=True),
    }
    frames = []
    print("\n%-12s %8s %8s %8s %8s %s" % ("setting", "macro01", "macroAUC", "min01", "max01",
                                          "n>=0.60"))
    for name, f in settings.items():
        f = f.assign(setting=name); frames.append(f)
        print("%-12s %8.4f %8.4f %8.4f %8.4f %d/%d"
              % (name, f.auc01.mean(), f.auc.mean(), f.auc01.min(), f.auc01.max(),
                 int((f.auc01 >= AUC_BEST).sum()), len(f)))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(RESULTS, "pairwise.csv"), index=False)

    cv, lo, ls = (settings[k].auc01.mean() for k in ("cv_grouped", "lopo", "lopo_strict"))
    print("\nreference points: best of 126 submissions %.2f | germline-only in-distribution %.4f"
          % (AUC_BEST, GERM_CV))
    print("in-distribution ceiling %.4f; blind (peptide unseen) %.4f; blind and receptor-"
          "disjoint %.4f" % (cv, lo, ls))

    macros = {
        "pwNpairs": "%d" % len(d), "pwNfeat": "%d" % X.shape[1],
        "pwCv": "%.3f" % cv, "pwLopo": "%.3f" % lo, "pwLopoStrict": "%.3f" % ls,
        "pwCvFull": "%.3f" % settings["cv_grouped"].auc.mean(),
        "pwLopoFull": "%.3f" % settings["lopo"].auc.mean(),
        "pwLopoStrictFull": "%.3f" % settings["lopo_strict"].auc.mean(),
        "pwCvNabove": "%d" % int((settings["cv_grouped"].auc01 >= AUC_BEST).sum()),
        "pwLopoNabove": "%d" % int((settings["lopo"].auc01 >= AUC_BEST).sum()),
        "pwLopoStrictNabove": "%d" % int((settings["lopo_strict"].auc01 >= AUC_BEST).sum()),
        "pwNtotal": "%d" % settings["cv_grouped"].shape[0],
        "pwAucBest": "%.2f" % AUC_BEST,
    }
    with open(os.path.join(ADAT, "pairwise_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "pairwise.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "pairwise_macros.tex"), len(macros)))


def demo():
    """Self-check, and a measurement of what receptor grouping actually does here.

    Comparison at MATCHED fold sizes and matched shuffling, so grouping is the only thing that
    differs -- comparing GroupKFold against a shuffled StratifiedKFold would confound grouping
    with fold composition. Two facts are pinned, both contrary to the expected leakage story
    and both consequences of every receptor being positive once and negative nine times:
    the ungrouped split scores LOWER, and the pure receptor-identity channel is at or below
    chance.
    """
    d = pd.read_csv(BENCH, sep="\t")
    peps = sorted(d.peptide.unique())[:4]
    d = d[d.peptide.isin(peps)].reset_index(drop=True)
    X, cat = features(d)
    assert not X[[c for c in X.columns if c not in cat]].isna().any().any()
    assert X.shape[1] == len(SEGMENTS) * (1 + len(AA) + len(PROPS)) + len(cat), X.shape

    groups = pd.Series(list(zip(d.tcra_cdr3, d.tcrb_cdr3)), index=d.index).astype(str)
    # grouping must leave no receptor on both sides of a fold
    for tr, te in GroupKFold(n_splits=3).split(X, d.label, groups):
        assert not (set(groups.iloc[tr]) & set(groups.iloc[te])), "receptor leaked across folds"

    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    def fit(splits):
        s = np.zeros(len(d))
        for tr, te in splits:
            m = _model(cat).fit(X.iloc[tr], d.label.iloc[tr])
            s[te] = m.predict_proba(X.iloc[te])[:, 1]
        return _macro(d, s).auc01.mean()

    grouped = fit(StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
                  .split(X, d.label, groups))
    ungrouped = fit(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, d.label))
    assert ungrouped < grouped, (ungrouped, grouped)

    # the pure receptor-identity channel: score a test receptor by its training positive-rate,
    # with no features at all. Uninformative by construction, so at or below chance.
    ident = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, d.label):
        rate = d.iloc[tr].groupby(groups.iloc[tr]).label.mean()
        s = groups.iloc[te].map(rate).fillna(rate.mean()).to_numpy()
        ident.append(roc_auc_score(d.label.iloc[te], s))
    assert max(ident) < 0.5, ident
    print("pairwise.demo OK  (%d features, no receptor leaks across grouped folds; at matched "
          "fold sizes ungrouped scores %.3f against grouped %.3f, and the pure "
          "receptor-identity channel is below chance at full AUC %.3f)"
          % (X.shape[1], ungrouped, grouped, float(np.mean(ident))))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
