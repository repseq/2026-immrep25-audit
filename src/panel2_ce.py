# Fig. 3 panel 2: multi-class epitope prediction from ESM-2 embeddings, cross-entropy on an 80/20
# train-test split, per chain.
# 2026-08-08
#
# Model: a perceptron with a softmax output (MLPClassifier, one hidden layer) fitted on the 480-d
# ESM-2 CDR3 embedding to predict the epitope. Read-out is cross-entropy (log loss) on train and on
# test, which is a proper scoring rule -- unlike AUC it is sensitive to calibration, and unlike the
# earlier train-vs-test AUC it does not saturate at 1.000 and so leaves a meaningful gap.
#
# Comparability across cohorts: raw cross-entropy is not comparable, because a cohort with more
# epitopes has a harder problem -- predicting the marginal gives ln K nats. So the headline quantity
# is the normalised information gain
#
#     G = 1 - CE / CE_chance,     CE_chance = entropy of the training label marginal (nats)
#
# G = 1 when the epitope is predicted with certainty, G = 0 when the model does no better than
# guessing the label frequencies, and G < 0 when it is worse than that. This divides out the epitope
# count exactly the way Q's normalisation does in panel 1.
#
# Split: plain random 80/20 over records, N_REP repeats, as requested. Per-chain de-duplication has
# already removed identical (epitope, cdr3) pairs, so no receptor is its own training copy.

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

import os
SCR = os.environ.get("PANEL1_WORK", "/Users/mikesh/tmp/fig3pc")   # per-chain frames + ESM embeddings
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ADAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "appendix", "analysis")
COHORTS = ["tcrvdb_true", "vdjdb_hq", "immrep22_true", "vdjdb_lq", "immrep25_pos", "pairseq_mock"]
TEST_FRAC = 0.20
N_REP = 5
HIDDEN = (64,)
MAX_ROWS = 30000      # cap only the two largest cohorts' rows, to keep the fit bounded; stated in report


def one(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, max_iter=400, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    lab = clf.classes_
    ce_tr = log_loss(ytr, clf.predict_proba(sc.transform(Xtr)), labels=lab)
    ce_te = log_loss(yte, clf.predict_proba(sc.transform(Xte)), labels=lab)
    # chance = entropy of the TRAINING label marginal, evaluated on the test labels
    _, cnt = np.unique(ytr, return_counts=True)
    pri = cnt / cnt.sum()
    pri_map = dict(zip(np.unique(ytr), pri))
    ce_ch = float(-np.mean([np.log(max(pri_map.get(v, 1e-12), 1e-12)) for v in yte]))
    return ce_tr, ce_te, ce_ch


def main():
    rows = []
    for ch in ("A", "B"):
        for coh in COHORTS:
            fr = pd.read_csv(f"{SCR}/frames/{coh}_{ch}.csv")
            X = np.load(f"{SCR}/emb/{coh}_{ch}.npy")
            y = fr.epitope.to_numpy()
            n0 = len(fr)
            if n0 > MAX_ROWS:
                rng = np.random.default_rng(0)
                keep = np.sort(rng.choice(n0, MAX_ROWS, replace=False))
                X, y = X[keep], y[keep]
            # drop epitopes left with <5 records so the stratified split is defined
            vc = pd.Series(y).value_counts()
            m = np.isin(y, vc[vc >= 5].index)
            X, y = X[m], y[m]
            res = np.array([one(X, y, s) for s in range(N_REP)])
            ce_tr, ce_te, ce_ch = res.mean(0)
            sd = res.std(0, ddof=1)
            g_tr, g_te = 1 - ce_tr / ce_ch, 1 - ce_te / ce_ch
            rows.append(dict(cohort=coh, chain=ch, n_used=len(y), n_full=n0,
                             n_epitopes=int(len(np.unique(y))),
                             ce_train=ce_tr, ce_test=ce_te, ce_chance=ce_ch,
                             ce_test_sd=sd[1], gain_train=g_tr, gain_test=g_te,
                             gain_test_sd=sd[1] / ce_ch, gap=ce_te - ce_tr))
            print(f"{coh:14} {ch}  n={len(y):6d} K={len(np.unique(y)):3d}  "
                  f"CE tr={ce_tr:.3f} te={ce_te:.3f} chance={ce_ch:.3f}  "
                  f"G_tr={g_tr:+.3f} G_te={g_te:+.3f}")
    df = pd.DataFrame(rows)
    for ch in ("A", "B"):
        m = df.chain == ch
        ref = float(df.loc[m & (df.cohort == "immrep25_pos"), "gain_test"].iloc[0])
        df.loc[m, "ratio_vs_immrep25"] = df.loc[m, "gain_test"] / ref if ref else np.nan
    df.to_csv(os.path.join(OUT, "panel2_ce.csv"), index=False)

    # macros for Fig. 3c,d and its caption. Emitted here so they cannot go stale when a cohort
    # changes -- they were previously written by hand, which is how the VDJdb values survived a
    # change of cohort definition.
    tag = {"tcrvdb_true": "Tt", "vdjdb_hq": "Hq", "immrep22_true": "Ii", "vdjdb_lq": "Lq",
           "immrep25_pos": "Imm", "pairseq_mock": "Ps", "airr_control": "Airr",
           "olga_random": "Olga"}
    macros = {"ceTestFrac": "%d" % round(100 * TEST_FRAC), "ceReps": "%d" % N_REP,
              "ceHidden": "%d" % HIDDEN[0], "ceMaxRows": "{:,}".format(MAX_ROWS),
              "ceMinPer": "5"}
    for _, r in df.iterrows():
        k = tag.get(r.cohort)
        if not k:
            continue
        macros["ceTr%s%s" % (k, r.chain)] = "%.2f" % r.ce_train
        macros["ceTe%s%s" % (k, r.chain)] = "%.2f" % r.ce_test
        macros["ceCh%s%s" % (k, r.chain)] = "%.2f" % r.ce_chance
        macros["ceG%s%s" % (k, r.chain)] = "%+.2f" % r.gain_test
    with open(os.path.join(ADAT, "panel2_ce_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote results/panel2_ce.csv + appendix/analysis/panel2_ce_macros.tex")


if __name__ == "__main__":
    main()
