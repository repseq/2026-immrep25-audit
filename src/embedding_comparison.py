#!/usr/bin/env python
# 2026-09-14  Do learned TCR representations clear the bound the benchmark's design imposes?
#
# Three reviewers answered this manuscript's probes with the same objection: a better method would
# find the signal. A bound answers that in principle; this module answers it in practice, by handing
# IMMREP25 to the strongest learned receptor representations available and scoring them under the
# challenge's own condition. The representations are cached by src/embed_cache.py (a separate,
# pinned environment -- see its header); nothing here re-embeds anything.
#
# ONE THING VARIES. The protocol is pairwise.py's, imported rather than reimplemented: the same
# HistGradientBoosting model, the same grouped five-fold split, the same leave-one-peptide-out
# arm with the held-out peptide's cognate receptors withheld, and the same per-peptide macro over
# McClish-standardised AUC_0.1. Only the feature block changes, so a difference between rows is a
# difference between representations and not between training protocols.
#
# THREE ARMS, answering three different objections:
#   blind  A single pre-specified scalar per receptor -- the first principal component of the
#          representation -- with no fitting and no peptide. negative_design.py proves any such
#          score is pinned at chance by the within-MHC negative construction, whatever it measures.
#          This is that theorem's empirical test, on representations built to separate receptors.
#   lopo   Leave-one-peptide-out, strict: neither the held-out peptide nor its binders are in
#          training. This is what the challenge entrants faced.
#   cv     Grouped five-fold with every peptide seen. An in-distribution ceiling, not a method
#          result -- reported so that a null in `lopo` cannot be read as the representation
#          being too weak to fit anything.
#
# WIDTH IS EQUALISED, NOT INHERITED. The representations range from 64 dimensions (sceptr) to 2560
# (ESM-2 650M, two chains concatenated), and a wider feature block changes what the model can fit
# irrespective of what the representation knows. Each is therefore projected onto its first N_PC
# principal components before fitting. The projection is unsupervised -- it never sees a label --
# which is the same argument esm_pca.py makes in this repo for the same reason.
#
# WHAT THIS CANNOT SHOW. A null here is a statement about what these representations extract from
# THIS benchmark, whose negatives are other epitopes' genuine binders. It is not a statement about
# the representations, which are not on trial; sceptr's published performance is on sets whose
# negatives are non-binders.
#
# Run: python src/embedding_comparison.py       (--demo for the self-check)
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
CACHE = os.path.join(REPO, "cache", "embed")
KEY = os.path.join(CACHE, "receptor_key.csv")

import pairwise as PW                                             # noqa: E402

MAX_FPR = 0.1
AUC_BEST = 0.60       # best of 126 IMMREP25 submissions (Richardson 2026)
N_PC = 64             # common representation width; sceptr's own, the narrowest on offer
SEED = 0
RECEPTOR = ["tcra_cdr3", "tcrb_cdr3", "tcra_v", "tcrb_v", "tcra_j", "tcrb_j"]

# How each cached representation is described in the paper. Ablations are the sceptr authors' own
# (Nagano et al., Cell Systems 2025), which is what makes them usable as a scale here.
LABEL = {
    "sceptr_default": "SCEPTR",
    "sceptr_large": "SCEPTR (large)",
    "sceptr_small": "SCEPTR (small)",
    "sceptr_cdr3_only": "SCEPTR (CDR3 only)",
    "sceptr_a_sceptr": "SCEPTR (alpha only)",
    "sceptr_b_sceptr": "SCEPTR (beta only)",
    "sceptr_blosum": "SCEPTR (BLOSUM ablation)",
    "sceptr_average_pooling": "SCEPTR (average pooling)",
    "sceptr_mlm_only": "SCEPTR (MLM only)",
    "sceptr_shuffled_data": "SCEPTR (shuffled-data control)",
    "sceptr_synthetic_data": "SCEPTR (synthetic-data control)",
    "esm2_35M": "ESM-2 35M",
    "esm2_650M": "ESM-2 650M",
    "tcrbert": "TCR-BERT",
}


def bench() -> pd.DataFrame:
    d = pd.read_csv(BENCH, sep="\t")
    assert len(d) == 10000 and int(d.label.sum()) == 1000, (len(d), int(d.label.sum()))
    return d


def load_reps() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """The receptor key and every cached representation, as {name: (n_receptors, d)}."""
    assert os.path.exists(KEY), "run .venv-embed/bin/python src/embed_cache.py first"
    key = pd.read_csv(KEY)
    reps = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "*.npy"))):
        name = os.path.splitext(os.path.basename(path))[0]
        X = np.load(path)
        assert X.shape[0] == len(key), (name, X.shape, len(key))
        reps[name] = X
    assert reps, "no representations cached in %s" % CACHE
    return key, reps


def project(X: np.ndarray, n: int = N_PC) -> np.ndarray:
    """Unsupervised projection onto a common width. Labels are never involved."""
    n = min(n, X.shape[1], X.shape[0])
    return PCA(n_components=n, random_state=SEED).fit_transform(X)


def row_index(d: pd.DataFrame, key: pd.DataFrame) -> np.ndarray:
    """For each of the 10000 labelled rows, the index of its receptor in the key."""
    pos = {t: i for i, t in enumerate(map(tuple, key[RECEPTOR].to_numpy()))}
    idx = np.array([pos[t] for t in map(tuple, d[RECEPTOR].to_numpy())])
    assert len(idx) == len(d) and idx.max() == len(key) - 1
    return idx


def peptide_side(d: pd.DataFrame) -> pd.DataFrame:
    """The peptide half of the design, taken from pairwise.features() rather than rewritten.

    THIS IS LOAD-BEARING, and omitting it is a silent tautology. A receptor occupies ten rows --
    its one positive and its nine same-allele negatives -- and without peptide-derived columns
    all ten carry identical features, so the model can only emit one score per receptor. By the
    rank identity in negative_design.py the macro is then EXACTLY 0.5 whatever the representation
    knows, and a table of 0.5000s would look like a finding while measuring nothing. Taking the
    columns from pairwise keeps the peptide side identical across representations, so the only
    thing that varies down the table is the receptor representation.
    """
    XA, _ = PW.features(d)
    cols = [c for c in XA.columns if c.startswith("peptide_")]
    assert cols, "pairwise.features() no longer emits peptide_* columns"
    P = XA[cols].copy()
    P["hla"] = d.hla.astype("category")
    return P


def design(d: pd.DataFrame, V: np.ndarray, P: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Receptor representation, peptide features, and HLA -- pairwise.py's only categorical."""
    X = pd.DataFrame(V, index=d.index, columns=["e%03d" % i for i in range(V.shape[1])])
    return pd.concat([X, P], axis=1), ["hla"]


def arms(d: pd.DataFrame, V: np.ndarray, P: pd.DataFrame) -> dict[str, pd.DataFrame]:
    X, cat = design(d, V, P)
    groups = pd.Series(list(zip(d.tcra_cdr3, d.tcrb_cdr3)), index=d.index).astype(str)
    return {"blind": PW._macro(d, V[:, 0]),                    # PC1, no fit, no peptide
            "lopo": PW.run_lopo(d, X, cat, strict=True),
            "cv": PW.run_cv(d, X, cat, groups)}


def main():
    d = bench()
    key, reps = load_reps()
    idx = row_index(d, key)

    P = peptide_side(d)                      # identical for every representation; computed once
    rows = []
    for name, X0 in reps.items():
        V = project(X0)[idx]
        for arm, res in arms(d, V, P).items():
            rows.append(dict(rep=name, label=LABEL.get(name, name), arm=arm,
                             dim_raw=X0.shape[1], dim_used=V.shape[1],
                             macro01=float(res.auc01.mean()), max01=float(res.auc01.max()),
                             macro_full=float(res.auc.mean()),
                             n_above_best=int((res.auc01 >= AUC_BEST).sum()),
                             n_ep=len(res)))
            print("  %-30s %-6s macro %.4f  max %.4f  >=%.2f in %d/%d"
                  % (LABEL.get(name, name), arm, rows[-1]["macro01"], rows[-1]["max01"],
                     AUC_BEST, rows[-1]["n_above_best"], rows[-1]["n_ep"]))
    r = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "embedding_comparison.csv")
    r.to_csv(out, index=False)
    print("\nwrote %s" % out)
    emit_macros(r)


def emit_macros(r: pd.DataFrame) -> dict:
    """Macros from the scored table.

    Split out from main() deliberately: the fits take ~20 minutes, while the macro set is a pure
    function of the CSV, so revising which numbers the paper names must not cost a recompute.
    `python src/embedding_comparison.py --macros` re-emits from the committed CSV.

    No macro is emitted for the challenge best here -- it is \\veAucBest, already cited throughout
    the manuscript. A second macro for one published quantity is how two values of it end up in
    one paper.
    """
    lo = r[r.arm == "lopo"]
    bl = r[r.arm == "blind"]
    cv = r[r.arm == "cv"]
    best_lo = lo.loc[lo.macro01.idxmax()]
    best_bl = bl.loc[bl.macro01.idxmax()]
    best_cv = cv.loc[cv.macro01.idxmax()]
    sc = r[r.rep == "sceptr_default"].set_index("arm")

    def one(rep: str, arm: str) -> float:
        s = r[(r.rep == rep) & (r.arm == arm)]
        assert len(s) == 1, "%s/%s matched %d rows" % (rep, arm, len(s))
        return float(s.macro01.iloc[0])

    macros = {
        "ecmNrep": "%d" % r.rep.nunique(),
        "ecmNpc": "%d" % N_PC,
        # the shipped model, all three arms
        "ecmSceptrBlind": "%.4f" % sc.loc["blind", "macro01"],
        "ecmSceptrLopo": "%.4f" % sc.loc["lopo", "macro01"],
        "ecmSceptrCv": "%.4f" % sc.loc["cv", "macro01"],
        # ranges across every representation
        "ecmBlindLo": "%.4f" % bl.macro01.min(),
        "ecmBlindHi": "%.4f" % bl.macro01.max(),
        "ecmBlindBest": LABEL.get(best_bl.rep, best_bl.rep),
        "ecmLopoLo": "%.4f" % lo.macro01.min(),
        "ecmLopoHi": "%.4f" % lo.macro01.max(),
        "ecmLopoBest": LABEL.get(best_lo.rep, best_lo.rep),
        "ecmCvLo": "%.4f" % cv.macro01.min(),
        "ecmCvHi": "%.4f" % cv.macro01.max(),
        "ecmCvBest": LABEL.get(best_cv.rep, best_cv.rep),
        # the sceptr authors' own negative controls: the internal scale of the comparison
        "ecmShufLopo": "%.4f" % one("sceptr_shuffled_data", "lopo"),
        "ecmShufCv": "%.4f" % one("sceptr_shuffled_data", "cv"),
        "ecmSynthLopo": "%.4f" % one("sceptr_synthetic_data", "lopo"),
        "ecmSynthCv": "%.4f" % one("sceptr_synthetic_data", "cv"),
        "ecmNclearBest": "%d" % int((lo.macro01 >= AUC_BEST).sum()),
        "ecmNclearBlind": "%d" % int((bl.macro01 >= AUC_BEST).sum()),
        # the two counts the result turns on: nothing transfers, and fitting is not the limit
        "ecmNbelowChance": "%d" % int((lo.macro01 < 0.5).sum()),
        "ecmNcvAboveBest": "%d" % int((cv.macro01 >= AUC_BEST).sum()),
    }
    path = os.path.join(ADAT, "embedding_comparison_macros.tex")
    with open(path, "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("wrote %s (%d macros)" % (path, len(macros)))
    return macros


def demo():
    """Self-check. The load-bearing assertions are that the representations are joined to the
    right rows, and that the blind arm reproduces the negative-design identity: the macro-mean
    FULL AUC of ANY receptor-level score is exactly K/2 summed over pools, because each receptor
    occupies one positive and nine negative rows of its own allele. If that identity fails, the
    join is wrong and every number in the table is meaningless."""
    d = bench()
    key, reps = load_reps()
    idx = row_index(d, key)

    # 1. the join is a bijection on receptors, and carries the labels it should
    assert len(np.unique(idx)) == len(key), "the join lost receptors"
    name, X0 = next(iter(reps.items()))
    V = project(X0)[idx]
    same = d.groupby(list(RECEPTOR)).ngroup()
    assert (pd.Series(idx).groupby(same).nunique() == 1).all(), "one receptor, two embeddings"

    # 2. each receptor is one positive and nine same-allele negatives -- the design the bound rests on
    per = d.groupby(RECEPTOR).label.agg(["sum", "size"])
    assert (per["sum"] == 1).all() and (per["size"] == 10).all(), "not the 1-in-10 design"

    # 3. a receptor-only score has macro-mean full AUC of exactly 1/2 per allele
    for hla, g in d.groupby("hla"):
        res = PW._macro(g, V[g.index.to_numpy(), 0])
        assert abs(res.auc.mean() - 0.5) < 1e-9, (hla, res.auc.mean())

    # 4. THE check, and the reason arm `cv` is not trivially 0.5: the fitted design must
    #    distinguish the ten rows a receptor occupies. If it cannot, every fitted arm returns
    #    exactly chance by the identity asserted just above, and the table measures nothing.
    P = peptide_side(d)
    X, _ = design(d, V, P)
    nun = len(X.drop(columns=["hla"]).round(6).drop_duplicates())
    assert nun > len(key), (
        "the design holds %d distinct rows for %d receptors: the peptide side is missing, so "
        "every fitted macro would be exactly 0.5 by construction" % (nun, len(key)))

    print("embedding_comparison.demo OK  (%d representations, %d receptors joined onto %d rows; "
          "every receptor 1 positive + 9 same-allele negatives; a receptor-only score gives "
          "macro full AUC 0.5 exactly on both alleles; the fitted design resolves %d distinct "
          "rows from %d receptors)" % (len(reps), len(key), len(d), nun, len(key)))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--macros" in sys.argv:            # re-emit from the committed CSV, no refitting
        emit_macros(pd.read_csv(os.path.join(RESULTS, "embedding_comparison.csv")))
    else:
        main()
