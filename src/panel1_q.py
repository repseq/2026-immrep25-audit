"""Fig. 3 panel 1: epitope recovery Q of the tcrdist3 neighbourhood clustering, per chain.
# 2026-08-08

Replaces the CV neighbour-voting yield (coverage x precision) used in earlier drafts. Q is the
normalised homogeneity--parsimony score of Tiffeau-Mayer (arXiv:2607.20799), computed with the
authors' reference implementation (`clustereval`):

    h = 1 - H(C|K)/H(C)                     homogeneity: do clusters predict the epitope
    p = 1 - H(K|C)/(log N - H(C))           parsimony:   without shattering each epitope
    Q = 2hp/(h+p)

Both scores are normalised by the maximum attainable given the epitope partition C, so cohort size
and epitope count divide out -- which is what makes cohorts carrying 2 to 84 epitopes comparable.
Purity/precision live in the paper's separate set-matching branch and are deliberately not used.

The clustering is UNSUPERVISED (epitope labels never build it), so there is no train/test split and
no group key: single-linkage at radius theta == connected components of the theta-thresholded
tcrdist graph. Identical CDR3s are already removed by the per-chain de-duplication (Methods).

theta = 18 is tcrdist3's own default (`TCRpublic.radius`, tcrdist/public.py). Q is stable over
theta = 10-18; at theta = 24 the pairSEQ mock floor rises to two thirds of VDJdb(HQ), so the loose
radius is not a usable operating point.

Read-outs per cohort per chain:
  * q_full  -- Q on the whole cohort
  * q80, ci -- mean and 95 percentile interval over 100 random 80% subsamples of the rows, drawn
               WITHOUT replacement. Not a with-replacement bootstrap: a duplicated receptor is
               simultaneously a same-cluster and same-epitope pair, i.e. exactly the configuration Q
               rewards, so resampling with replacement manufactures structure (it puts Q ~ 0.25 even
               on random labels). Q grows with depth, so q80 sits slightly below q_full and the
               cohort comparison is made draw-paired at matched 80% depth.
  * p_perm  -- label-permutation test: epitopes shuffled, clustering held fixed. Answers whether the
               cohort carries any recoverable epitope structure; this is what the figure's stars show.

Inputs: per-chain frames + tcrdist3 radius-24 edge lists (see SOURCES.md).
Usage: python src/panel1_q.py    # writes results/panel1_q.csv + appendix/analysis/panel1_q_macros.tex
"""
from __future__ import annotations
import os
import importlib.util
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

# staging dir holding the per-chain frames and tcrdist3 edge lists (see SOURCES.md)
WORK = os.environ.get("PANEL1_WORK", "/Users/mikesh/tmp/fig3pc")
CE = os.environ.get("CLUSTEREVAL", "/Users/mikesh/tmp/clustereval/clustereval/core.py")

COHORTS = ["tcrvdb_true", "vdjdb_hq", "immrep22_true", "vdjdb_lq", "immrep25_pos",
           "pairseq_mock", "airr_control", "olga_random"]
TAG = {"tcrvdb_true": "Tt", "vdjdb_hq": "Hq", "immrep22_true": "Ii", "vdjdb_lq": "Lq",
       "immrep25_pos": "Imm", "pairseq_mock": "Ps", "airr_control": "Airr", "olga_random": "Olga"}
THETA = 18            # tcrdist3 default (TCRpublic.radius)
FRAC, N_DRAW, N_PERM = 0.80, 100, 1000
REF = "immrep25_pos"

# clustereval's __init__ calls importlib.metadata.version(), which needs the package installed;
# load core.py by path so the analysis venv is untouched.
_spec = importlib.util.spec_from_file_location("ce_core", CE)
_ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ce)


def q_of(y, k):
    h = _ce.homogeneity_score(y, k)
    p = _ce.parsimony_score(y, k)
    return (0.0 if (h + p) == 0 else 2 * h * p / (h + p)), h, p


def comps(n, ed2, idx=None):
    """Single-linkage clusters = connected components of the thresholded graph, optionally on idx."""
    if idx is None:
        idx = np.arange(n)
    pos = -np.ones(n, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    e = ed2[(pos[ed2[:, 0]] >= 0) & (pos[ed2[:, 1]] >= 0)]
    nn = len(idx)
    if len(e) == 0:
        return np.arange(nn)
    g = coo_matrix((np.ones(len(e)), (pos[e[:, 0]], pos[e[:, 1]])), shape=(nn, nn))
    return connected_components(g, directed=False)[1]


def run():
    rng = np.random.default_rng(0)
    rows, draws = [], {}
    for ch in ("A", "B"):
        for coh in COHORTS:
            fr = pd.read_csv(os.path.join(WORK, "frames", f"{coh}_{ch}.csv"))
            ed = pd.read_csv(os.path.join(WORK, "td", f"edges_{coh}_{ch}.tsv"),
                             sep="\t").to_numpy()
            ed2 = ed[ed[:, 2] <= THETA][:, :2].astype(np.int64)
            y = fr.epitope.to_numpy()
            n = len(fr)
            lab = comps(n, ed2)
            q, h, p = q_of(y, lab)
            m = max(2, int(round(FRAC * n)))
            sub = np.array([q_of(y[i], comps(n, ed2, i))[0]
                            for i in (np.sort(rng.choice(n, m, replace=False))
                                      for _ in range(N_DRAW))])
            draws[(coh, ch)] = sub
            null = np.array([q_of(rng.permutation(y), lab)[0] for _ in range(N_PERM)])
            rows.append(dict(cohort=coh, chain=ch, theta=THETA, n=n, n_sub=m,
                             n_epitopes=int(len(np.unique(y))), n_clusters=int(len(np.unique(lab))),
                             q_full=q, h=h, p=p,
                             q80=sub.mean(), q80_sd=sub.std(ddof=1),
                             ci_lo=float(np.percentile(sub, 2.5)),
                             ci_hi=float(np.percentile(sub, 97.5)),
                             q_null=null.mean(),
                             p_perm=(1 + int((null >= q).sum())) / (N_PERM + 1)))
            print("%-14s %s  Q=%.4f  Q80=%.4f [%.4f,%.4f]  P_perm=%.4f"
                  % (coh, ch, q, sub.mean(), rows[-1]["ci_lo"], rows[-1]["ci_hi"],
                     rows[-1]["p_perm"]))
    df = pd.DataFrame(rows)
    for ch in ("A", "B"):
        m = df.chain == ch
        ref = float(df.loc[m & (df.cohort == REF), "q80"].iloc[0])
        df.loc[m, "ratio_vs_immrep25"] = df.loc[m, "q80"] / ref
    # paired-by-draw comparison against IMMREP25 (same draw index, matched 80% depth)
    for ch in ("A", "B"):
        ref = np.asarray(draws[(REF, ch)])
        for i, r in df.iterrows():
            if r.chain != ch or r.cohort == REF:
                continue
            d = np.asarray(draws[(r.cohort, ch)]) - ref
            f = float((d > 0).mean())
            df.loc[i, "frac_gt"] = f
            df.loc[i, "p_two"] = max(min(1.0, 2 * min(f, 1 - f)), 1.0 / (N_DRAW + 1))
    df.to_csv(os.path.join(RESULTS, "panel1_q.csv"), index=False)

    with open(os.path.join(ADAT, "panel1_q_macros.tex"), "w") as fh:
        fh.write("\\newcommand{\\qTheta}{%d}\n" % THETA)
        fh.write("\\newcommand{\\qFrac}{%d}\n" % int(FRAC * 100))
        fh.write("\\newcommand{\\qDraws}{%d}\n" % N_DRAW)
        fh.write("\\newcommand{\\qPerm}{%d}\n" % N_PERM)
        for _, r in df.iterrows():
            t = TAG[r.cohort] + r.chain
            fh.write("\\newcommand{\\q%s}{%.3f}\n" % (t, r.q80))
            fh.write("\\newcommand{\\qRat%s}{%.1f}\n" % (t, r.ratio_vs_immrep25))
            fh.write("\\newcommand{\\qEp%s}{%d}\n" % (t, r.n_epitopes))
    print("\nwrote results/panel1_q.csv + appendix/analysis/panel1_q_macros.tex")


if __name__ == "__main__":
    run()
