#!/usr/bin/env python
# 2026-09-13  Does "structural understanding" reduce to a similarity lookup?
#
# The IMMREP25 organisers report that the top-performing submissions incorporated structural
# modelling, and read that as evidence a structural understanding aids unseen-peptide
# prediction. Every one of those pipelines scores a pair by FOLDING CONFIDENCE -- PAE for
# TCRdock, interface ipTM for TCRmodel2, normalised pLDDT for the AlphaFold3 entry -- and a
# folding model's confidence is, in part, a statement about how well the input resembles what
# is already in the PDB. So the question is whether the benchmark can be scored by the
# resemblance alone, with no folding and no physics.
#
# This is also the one channel the learnability bound (Result 3) leaves open. A predicted
# complex is a function of its input sequence plus a FIXED reference database, so the only
# route by which structure can carry antigen information the sequence lacks is peptide-keyed
# template retrieval. The manuscript argues that channel is empirically closed for IMMREP25,
# whose twenty peptides have no template coverage; a reviewer objected that this was asserted
# rather than measured. Here it is measured directly.
#
# Score: for a candidate (TCR, peptide) pair, the best BLOSUM62 local-alignment similarity to
# any reference TCR-pMHC complex, summing the CDR3a, CDR3b and peptide terms. Two references:
#   * solved crystal structures -- the tcren Native2026 markup (human complexes), i.e. exactly
#     the template set a folding model could retrieve from;
#   * VDJdb paired complexes -- a far larger, sequence-only reference, stratified by epitope.
# A TCR-only variant drops the peptide term, which reduces the score to publicity and
# isolates how much of any signal is peptide-keyed at all.
#
# Run: python src/retrieval_baseline.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
CRYSTALS = os.path.expanduser("~/vcs/code/tcren/scratch/markup_2026.csv")

MAX_FPR = 0.1
AUC_BEST = 0.60        # best of 126 IMMREP25 submissions (Richardson 2026)
VDJDB_CAP = 5000       # reference complexes, stratified by epitope; stated, not silent
SEED = 0


def _aligner():
    al = PairwiseAligner(scoring="blastp")
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.mode = "local"
    return al


def simmat(queries, refs, al=None) -> np.ndarray:
    """Pairwise BLOSUM62 local-alignment scores, queries x refs."""
    al = al or _aligner()
    M = np.zeros((len(queries), len(refs)), dtype=np.float32)
    for i, q in enumerate(queries):
        for j, r in enumerate(refs):
            M[i, j] = al.score(q, r)
    return M


def retrieval_scores(rec: pd.DataFrame, ref: pd.DataFrame, peptides):
    """Best-matching-reference score for every (receptor, peptide) pair.

    The maximum over references of (sim_a + sim_b + sim_pep) does not factorise, but the three
    similarity matrices do: compute them once and broadcast, which turns ~10^7 alignments into
    ~10^6 plus a cheap numpy maximum.
    """
    al = _aligner()
    A = simmat(rec.tcra_cdr3.tolist(), ref.cdr3a.tolist(), al)
    B = simmat(rec.tcrb_cdr3.tolist(), ref.cdr3b.tolist(), al)
    P = simmat(list(peptides), ref.epitope.tolist(), al)
    AB = A + B
    S = np.empty((len(rec), len(peptides)), dtype=np.float32)
    for k in range(len(peptides)):
        S[:, k] = (AB + P[k][None, :]).max(axis=1)
    return S, AB.max(axis=1)          # (pair score, TCR-only score)


def _crystal_reference() -> pd.DataFrame:
    cry = pd.read_csv(CRYSTALS)
    cry = cry[cry.species.astype(str).str.contains("Human", case=False, na=False)]
    cry = cry.dropna(subset=["cdr3a", "cdr3b", "peptide"]).rename(columns={"peptide": "epitope"})
    return cry[["cdr3a", "cdr3b", "epitope"]].drop_duplicates().reset_index(drop=True)


def _vdjdb_reference() -> pd.DataFrame:
    from load_data import load_vdjdb
    _, vp = load_vdjdb()
    vp = vp.dropna(subset=["cdr3a", "cdr3b", "epitope"]).drop_duplicates(["cdr3a", "cdr3b", "epitope"])
    per_ep = max(1, VDJDB_CAP // vp.epitope.nunique())
    # shuffle+head rather than groupby.apply: pandas 3 drops the grouping column in apply
    vp = (vp.sample(frac=1.0, random_state=SEED)
            .groupby("epitope", group_keys=False).head(per_ep).reset_index(drop=True))
    if len(vp) > VDJDB_CAP:
        vp = vp.sample(VDJDB_CAP, random_state=SEED).reset_index(drop=True)
    return vp[["cdr3a", "cdr3b", "epitope"]]


def score_benchmark(ref: pd.DataFrame, label: str) -> pd.DataFrame:
    bench = pd.read_csv(BENCH, sep="\t")
    recept = bench[["tcra_cdr3", "tcrb_cdr3"]].drop_duplicates().reset_index(drop=True)
    peps = sorted(bench.peptide.unique())
    S, tcr_only = retrieval_scores(recept, ref, peps)
    ridx = {(a, b): i for i, (a, b) in enumerate(zip(recept.tcra_cdr3, recept.tcrb_cdr3))}
    r = np.array([ridx[(a, b)] for a, b in zip(bench.tcra_cdr3, bench.tcrb_cdr3)])
    pk = {p: k for k, p in enumerate(peps)}
    bench["pair"] = S[r, [pk[p] for p in bench.peptide]]
    bench["tcr_only"] = tcr_only[r]
    rows = []
    for p in peps:
        sub = bench[bench.peptide == p]
        rows.append(dict(reference=label, peptide=p, hla=sub.hla.iloc[0],
                         n_ref=len(ref), n_ref_ep=int(ref.epitope.nunique()),
                         auc01=roc_auc_score(sub.label, sub.pair, max_fpr=MAX_FPR),
                         auc=roc_auc_score(sub.label, sub.pair),
                         auc01_tcr_only=roc_auc_score(sub.label, sub.tcr_only, max_fpr=MAX_FPR)))
    return pd.DataFrame(rows)


def main():
    frames = []
    for ref, label in ((_crystal_reference(), "crystals"), (_vdjdb_reference(), "vdjdb")):
        f = score_benchmark(ref, label)
        frames.append(f)
        print("\n=== reference: %s (%d complexes, %d epitopes) ==="
              % (label, f.n_ref.iloc[0], f.n_ref_ep.iloc[0]))
        print("  macro-AUC0.1 %.4f | macro full AUC %.4f | max %.4f | peptides >= %.2f: %d/%d"
              % (f.auc01.mean(), f.auc.mean(), f.auc01.max(), AUC_BEST,
                 int((f.auc01 >= AUC_BEST).sum()), len(f)))
        print("  TCR-only (publicity, no peptide term): macro %.4f | max %.4f | >= %.2f: %d/%d"
              % (f.auc01_tcr_only.mean(), f.auc01_tcr_only.max(), AUC_BEST,
                 int((f.auc01_tcr_only >= AUC_BEST).sum()), len(f)))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(RESULTS, "retrieval_baseline.csv"), index=False)

    macros = {}
    for label, tag in (("crystals", "Cry"), ("vdjdb", "Vdj")):
        f = out[out.reference == label]
        macros.update({
            f"retr{tag}Nref": "%d" % f.n_ref.iloc[0],
            f"retr{tag}Nep": "%d" % f.n_ref_ep.iloc[0],
            f"retr{tag}Macro": "%.3f" % f.auc01.mean(),
            f"retr{tag}Full": "%.3f" % f.auc.mean(),
            f"retr{tag}Max": "%.3f" % f.auc01.max(),
            f"retr{tag}Nabove": "%d" % int((f.auc01 >= AUC_BEST).sum()),
            f"retr{tag}TcrMacro": "%.3f" % f.auc01_tcr_only.mean(),
        })
    macros["retrNtotal"] = "%d" % out.peptide.nunique()
    macros["retrAucBest"] = "%.2f" % AUC_BEST
    with open(os.path.join(ADAT, "retrieval_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("\nwrote %s" % os.path.join(RESULTS, "retrieval_baseline.csv"))
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "retrieval_macros.tex"), len(macros)))


def demo():
    """Self-check, and the probe-power control the null needs.

    A null result from a similarity scorer is only informative if the scorer can detect
    similarity at all. We build a reference out of the benchmark's OWN positive pairs for
    half the peptides and confirm that retrieval then recovers those peptides strongly -- so
    the background reading against real references is a property of the data, not of the code.
    """
    bench = pd.read_csv(BENCH, sep="\t")
    peps = sorted(bench.peptide.unique())
    seeded = peps[:10]
    pos = bench[(bench.label == 1) & (bench.peptide.isin(seeded))]
    ref = (pos[["tcra_cdr3", "tcrb_cdr3", "peptide"]]
           .rename(columns={"tcra_cdr3": "cdr3a", "tcrb_cdr3": "cdr3b", "peptide": "epitope"})
           .drop_duplicates().reset_index(drop=True))
    f = score_benchmark(ref, "self")
    seen = f[f.peptide.isin(seeded)].auc01
    unseen = f[~f.peptide.isin(seeded)].auc01
    assert seen.mean() > 0.9, seen.mean()          # its own pairs must be recovered
    assert unseen.mean() < 0.6, unseen.mean()      # and the others must not be
    # the pair score must dominate the TCR-only score when the peptide term is informative
    assert f[f.peptide.isin(seeded)].auc01.mean() > f[f.peptide.isin(seeded)].auc01_tcr_only.mean()
    print("retrieval_baseline.demo OK  (self-reference recovers its own %d peptides at "
          "macro-AUC0.1 %.3f while the held-out %d sit at %.3f -- the probe has power)"
          % (len(seeded), seen.mean(), len(unseen), unseen.mean()))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
