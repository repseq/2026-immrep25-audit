"""Degenerate-learning probe: does a model trained on a cohort's epitope labels acquire a
generalizable recognition mechanism, or only memorize its training TCRs?
# 2026-08-06  (round-3 MAIN result)

For each cohort we take the cached ESM-2 embeddings (same subsample as esm_signal/esm_cluster),
DEDUPLICATE identical CDR3s (so held-out TCRs are never exact copies of training ones), and run a
within-epitope generalization test: a multinomial linear probe is fit on a training split to
predict the epitope from the embedding, and evaluated on held-out TCRs of the SAME epitopes.

Read-outs (mean over repeated stratified splits):
  * held-out macro one-vs-rest AUC  -- generalization (chance = 0.5)
  * memorization gap = train AUC - held-out AUC  -- how much is pure fitting
  * I_train vs I_heldout in bits (mi.mi_report on true-vs-predicted labels) -- the generalization
    face of the coarse-grained MI: memorization is always possible (I_train high), but only a real
    recognition mechanism carries I_heldout > 0.

Prediction: genuinely epitope-specific sets (VDJdb HQ, TCRvdb true, IMMREP22) generalize
(held-out AUC >> 0.5, small gap, I_heldout > 0); IMMREP25 memorizes (held-out AUC ~ 0.5, large
gap, I_heldout ~ 0). Cite Grazioli 2022, Lu 2025.

Usage: uv run python src/degenerate.py   # writes results/degenerate.csv + .dat + macros
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META      # noqa: E402
from esm_signal import _subsample, CACHE                        # noqa: E402
from mi import mi_report                                        # noqa: E402

from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold             # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
MIN_PER = 15          # min TCRs/epitope to qualify (guaranteed by esm subsample MIN_N=30)
N_REPEATS = 5         # grouped CV folds within one repeat
N_CVREP = 5           # independent repeats of the whole 5-fold CV, with a fresh shuffle each time.
                      # Folds of a single CV share training data, so their spread understates the
                      # uncertainty; the reported sd is BETWEEN REPEATS of the entire procedure.
N_PERM = 500          # permutation null for MI-in-bits


def _macro_auc(clf, Xs, y, max_fpr=None):
    """Macro one-vs-rest AUC, averaged over classes present in y (chance = 0.5).

    max_fpr=0.1 gives the McClish-standardised partial AUC over FPR<=0.1, i.e. the
    benchmark's own macro-AUC_0.1 metric (also chance = 0.5). Full AUC is dominated by
    the high-FPR region and compresses the cohorts together; AUC_0.1 scores the
    low-false-positive regime a specificity predictor is actually judged on."""
    proba = clf.predict_proba(Xs)
    yset = set(y)
    aucs = []
    for i, c in enumerate(clf.classes_):
        if c not in yset:
            continue
        pos = (y == c).astype(int)
        if 0 < pos.sum() < len(pos):
            aucs.append(roc_auc_score(pos, proba[:, i], max_fpr=max_fpr))
    return float(np.mean(aucs)) if aucs else np.nan


def _one_cohort(X: np.ndarray, epi: np.ndarray, seqs: np.ndarray) -> dict | None:
    # Keep convergent (identical) CDR3s as legitimate signal, but never let the SAME sequence
    # straddle the split: group by CDR3 so held-out TCRs are novel sequences of seen epitopes.
    vc = pd.Series(epi).value_counts()
    good = vc[vc >= MIN_PER].index
    m = np.isin(epi, good)
    X, epi, seqs = X[m], epi[m], seqs[m]
    if len(np.unique(epi)) < 2 or len(X) < 3 * MIN_PER:
        return None
    # Each repeat is a complete 5-fold grouped CV under a fresh shuffle; we average over its folds
    # and then report the spread BETWEEN repeats, which does not share training data.
    keys = ("tr_auc", "ho_auc", "tr_auc01", "ho_auc01", "it_bits", "ih_bits")
    rep_means = {k: [] for k in keys}
    for rep in range(N_CVREP):
        fold = {k: [] for k in keys}
        sgk = StratifiedGroupKFold(n_splits=N_REPEATS, shuffle=True, random_state=rep)
        for r, (tri, tei) in enumerate(sgk.split(X, epi, groups=seqs)):
            if len(set(epi[tri])) < 2 or len(set(epi[tei])) < 2:
                continue
            sc = StandardScaler().fit(X[tri])
            Xtr, Xte = sc.transform(X[tri]), sc.transform(X[tei])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, epi[tri])
            fold["tr_auc"].append(_macro_auc(clf, Xtr, epi[tri]))
            fold["ho_auc"].append(_macro_auc(clf, Xte, epi[tei]))
            fold["tr_auc01"].append(_macro_auc(clf, Xtr, epi[tri], max_fpr=0.1))
            fold["ho_auc01"].append(_macro_auc(clf, Xte, epi[tei], max_fpr=0.1))
            seed = rep * N_REPEATS + r
            fold["it_bits"].append(mi_report(epi[tri], clf.predict(Xtr), n_perm=N_PERM, seed=seed)["excess"])
            fold["ih_bits"].append(mi_report(epi[tei], clf.predict(Xte), n_perm=N_PERM, seed=seed)["excess"])
        if not fold["ho_auc"]:
            continue
        for k in keys:
            rep_means[k].append(float(np.mean(fold[k])))
    if not rep_means["ho_auc"]:
        return None
    tr_auc, ho_auc = np.array(rep_means["tr_auc"]), np.array(rep_means["ho_auc"])
    tr_auc01, ho_auc01 = np.array(rep_means["tr_auc01"]), np.array(rep_means["ho_auc01"])
    it_bits, ih_bits = np.array(rep_means["it_bits"]), np.array(rep_means["ih_bits"])
    Ha = float(np.log2(len(np.unique(epi))))
    return {"n": int(len(X)), "n_ep": int(len(np.unique(epi))), "H_a": Ha,
            "train_auc": float(tr_auc.mean()), "heldout_auc": float(ho_auc.mean()),
            "heldout_sd": float(ho_auc.std(ddof=1)),
            "train_auc01": float(tr_auc01.mean()), "heldout_auc01": float(ho_auc01.mean()),
            "heldout01_sd": float(ho_auc01.std(ddof=1)),
            "gap": float((tr_auc - ho_auc).mean()),
            "gap01": float((tr_auc01 - ho_auc01).mean()),
            "I_train": float(it_bits.mean()), "I_heldout": float(ih_bits.mean()),
            "I_heldout_sd": float(ih_bits.std(ddof=1)),
            "U_heldout": float(ih_bits.mean() / Ha) if Ha > 0 else 0.0}


CURVE_COHORTS = ["tcrvdb_true", "vdjdb_hq", "immrep22_true", "vdjdb_lq", "immrep25_pos",
                 "pairseq_mock"]   # the calibration ladder plus its platform-matched reference
CURVE_FRACS = (0.15, 0.3, 0.5, 0.7, 1.0)


def _learning_curve(X, epi, seqs):
    """Held-out read-outs against the FRACTION of a cohort's receptors used for training.

    Uses the same splitter as _one_cohort -- StratifiedGroupKFold(N_REPEATS) on the CDR3, so one
    fold is held out, it never shares a CDR3 with the training folds, and the held-out share is the
    1/N_REPEATS used everywhere else rather than a separate 30%. Averaged over the N_REPEATS choices
    of held-out fold. The training set is then thinned to a fraction of the available CDR3 groups.

    x is n_train / n_cohort, so cohorts spanning 120 to 4500 receptors are read on one axis; it
    cannot exceed (N_REPEATS-1)/N_REPEATS, the share available for training at all.
    """
    vc = pd.Series(epi).value_counts(); good = vc[vc >= MIN_PER].index
    m = np.isin(epi, good); X, epi, seqs = X[m], epi[m], seqs[m]
    if len(np.unique(epi)) < 2 or len(X) < 4 * MIN_PER:
        return []
    groups = pd.Categorical(seqs).codes.astype(np.int64)
    n_tot = len(X)
    out = []
    for frac in CURVE_FRACS:
        aucs, mis, top1, ntr = [], [], [], []
        sgk = StratifiedGroupKFold(n_splits=N_REPEATS, shuffle=True, random_state=0)
        for k, (tri, tei) in enumerate(sgk.split(X, epi, groups=groups)):
            trg = np.unique(groups[tri])
            rng = np.random.default_rng(100 + k)
            keep = rng.choice(trg, max(2, int(frac * len(trg))), replace=False)
            tri2 = tri[np.isin(groups[tri], keep)]
            if len(set(epi[tri2])) < 2 or len(set(epi[tei])) < 2:
                continue
            sc = StandardScaler().fit(X[tri2])
            Xte = sc.transform(X[tei])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tri2]), epi[tri2])
            aucs.append(_macro_auc(clf, Xte, epi[tei]))
            pred = clf.predict(Xte)
            mis.append(mi_report(epi[tei], pred, n_perm=200)["excess"])
            # top-1 accuracy in excess of chance, as a fraction of the attainable excess: a probe
            # that only learns to REJECT classes can lift the one-vs-rest AUC while leaving this at
            # zero, so the two read together separate rejection from assignment.
            s = len(np.unique(epi[tei]))
            top1.append((float((pred == epi[tei]).mean()) - 1.0 / s) / (1.0 - 1.0 / s))
            ntr.append(len(tri2))
        if aucs:
            out.append((frac, int(np.mean(ntr)), float(np.mean(ntr)) / n_tot,
                        float(np.mean(aucs)), float(np.mean(mis)), float(np.mean(top1))))
    return out


def run():
    rng = np.random.default_rng(0)
    coh = build_cohorts(include_olga=True)
    order = [c for c in HIERARCHY if c in coh]
    rows = []
    curve_rows = []
    for name in order:
        long = coh[name]["long"]
        for chain in ("A", "B"):
            if name == "mlr_prolif" and chain == "A":
                continue
            sub = _subsample(long[long.chain == chain][["epitope", "cdr3"]].dropna(), rng)
            if sub.epitope.nunique() < 2:
                continue
            cache = os.path.join(CACHE, f"{name}_{chain}.npy")
            if not os.path.exists(cache):
                continue
            X = np.load(cache)
            if X.shape[0] != len(sub):
                print("%-16s TR%s  SKIP (cache/len mismatch %d!=%d)" % (name, chain, X.shape[0], len(sub)))
                continue
            res = _one_cohort(X, sub.epitope.to_numpy(), sub.cdr3.to_numpy())
            if res is None:
                continue
            rows.append({"cohort": name, "label": COHORT_META[name]["label"], "chain": chain,
                         "expect": COHORT_META[name]["expect"], **res})
            print("%-16s TR%s  heldout_AUC=%.3f(gap %.3f)  I_train=%.3f I_heldout=%.3f bits  n=%d ep=%d"
                  % (name, chain, res["heldout_auc"], res["gap"], res["I_train"],
                     res["I_heldout"], res["n"], res["n_ep"]))
            if name in CURVE_COHORTS:
                for frac, ntr, fr_rec, ho, mi_b, t1 in _learning_curve(
                        X, sub.epitope.to_numpy(), sub.cdr3.to_numpy()):
                    curve_rows.append({"cohort": name, "label": COHORT_META[name]["label"],
                                       "chain": chain, "frac": frac, "n_train": ntr,
                                       "frac_receptors": fr_rec, "heldout_auc": ho,
                                       "I_heldout": mi_b, "top1_excess": t1})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "degenerate.csv"), index=False)

    # learning curve: held-out transfer vs training-set size (P3)
    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv(os.path.join(RESULTS, "degenerate_curve.csv"), index=False)
    with open(os.path.join(ADAT, "degenerate_curve.dat"), "w") as fh:
        fh.write("# cohort chain n_train frac_receptors heldout_auc I_heldout top1_excess frac\n")
        for _, r in cdf.iterrows():
            fh.write("%s %s %d %.4f %.4f %.4f %.4f %.3f\n"
                     % (r.cohort, r.chain, r.n_train, r.frac_receptors, r.heldout_auc,
                        r.I_heldout, r.top1_excess, r.frac))

    idx = {n: i for i, n in enumerate(HIERARCHY)}
    for chain in ("B", "A"):
        d = df[df.chain == chain].copy(); d["x"] = [idx[c] for c in d.cohort]
        with open(os.path.join(ADAT, f"degenerate_{chain}.dat"), "w") as fh:
            fh.write("# x cohort heldout_auc heldout_sd gap I_train I_heldout I_heldout_sd\n")
            d[["x", "cohort", "heldout_auc", "heldout_sd", "gap",
               "I_train", "I_heldout", "I_heldout_sd"]].to_csv(fh, sep=" ", index=False, header=False)

    # macros: IMMREP25 vs the seen benchmark IMMREP22 and the gold VDJdb HQ, TCRβ headline
    macros = {}
    def g(name, chain, col):
        r = df[(df.cohort == name) & (df.chain == chain)]
        return r[col].iloc[0] if len(r) else np.nan
    for tag, name in (("Imm", "immrep25_pos"), ("Ii", "immrep22_true"),
                      ("Hq", "vdjdb_hq"), ("Tt", "tcrvdb_true"),
                      ("Lq", "vdjdb_lq"), ("Ps", "pairseq_mock"),
                      ("Airr", "airr_control"), ("Olga", "olga_random")):
        macros["degHo" + tag] = "%.2f" % g(name, "B", "heldout_auc")
        macros["degHoPa" + tag] = "%.2f" % g(name, "B", "heldout_auc01")
        macros["degGapPa" + tag] = "%.2f" % g(name, "B", "gap01")
        macros["degGap" + tag] = "%.2f" % g(name, "B", "gap")
        macros["degIh" + tag] = "%.2f" % g(name, "B", "I_heldout")
        macros["degIt" + tag] = "%.2f" % g(name, "B", "I_train")
        macros["degU" + tag] = "%.3f" % g(name, "B", "U_heldout")
    macros["degMinPer"] = "%d" % MIN_PER
    macros["degRepeats"] = "%d" % N_REPEATS

    # Learning curve read at equal RELATIVE training size (the fraction of a cohort's own
    # receptors). Two things the absolute-count axis cannot show: where a cohort lands when all of
    # its trainable receptors are used, and whether its distance from the seen-peptide benchmark
    # narrows as receptors are added.
    def c(name, chain, col, frac):
        r = cdf[(cdf.cohort == name) & (cdf.chain == chain) & (cdf.frac == frac)]
        return float(r[col].iloc[0]) if len(r) else np.nan
    lo, hi = min(CURVE_FRACS), max(CURVE_FRACS)
    for tag, name in (("Imm", "immrep25_pos"), ("Ii", "immrep22_true"), ("Hq", "vdjdb_hq"),
                      ("Lq", "vdjdb_lq"), ("Tt", "tcrvdb_true"), ("Ps", "pairseq_mock")):
        for chain in ("A", "B"):
            macros["degCvHo%s%s" % (tag, chain)] = "%.2f" % c(name, chain, "heldout_auc", hi)
            macros["degCvIh%s%s" % (tag, chain)] = "%.2f" % c(name, chain, "I_heldout", hi)
    for chain in ("A", "B"):
        for lab, f in (("Lo", lo), ("Hi", hi)):
            d_auc = (c("immrep22_true", chain, "heldout_auc", f)
                     - c("immrep25_pos", chain, "heldout_auc", f))
            macros["degCvD%s%s" % (lab, chain)] = "%.2f" % d_auc
        macros["degCvShare%s" % chain] = "%.0f" % (100 * c("immrep25_pos", chain,
                                                          "frac_receptors", hi))
    with open(os.path.join(ADAT, "degenerate_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/degenerate.csv + degenerate_{A,B}.dat + degenerate_macros.tex")


if __name__ == "__main__":
    run()
