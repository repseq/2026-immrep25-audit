"""Build the pairSEQ mock-benchmark cohort (`pairseq_mock`) -- 2026-08-08.

A mock benchmark assembled from data that provably contains no epitope-specific signal,
produced on the *same commercial platform*, and by the *same selection logic*, as IMMREP25.

IMMREP25's beta labels come from a MIRA assay: T cells are distributed over wells, each well
receives a peptide pool, and a TCR is called specific for that pool when it is enriched in
the well relative to the others. Alpha chains are added afterwards by pairSEQ, which pairs
chains from the wells they jointly occupy.

In the pairSEQ experiment of Howie et al. (2015) the same plate format is used but *no
peptide is added*: a blood sample is split into 96 aliquots and cells are allocated to wells
at random. Well-to-well abundance differences are therefore pure sampling noise. Running
MIRA's selection rule over that plate produces exactly what MIRA would produce from a
peptide with no responding T cells at all.

We reproduce IMMREP25 field for field:

    epitope <- the well            (as a peptide pool is one well)
    HLA     <- the subject, X or Y (as a peptide's restricting allele)
    records <- the 50 TCRA and 50 TCRB most enriched in that well, each required to be
               >= ENRICH-fold more frequent in the well than its mean frequency across the
               other wells of the same subject -- MIRA's enrichment call, applied to noise
    pairing <- alpha and beta drawn at random within the well, never across wells or
               subjects (as IMMREP25's chains are matched post hoc, not per cell)

giving WELLS_PER_SUBJ x 2 labels x 50 = 1000 paired mock receptors, as in IMMREP25.

The enrichment filter is what makes the imitation faithful and also what keeps the cohort
clean: clonotypes abundant everywhere (which top every well's raw count list and would
otherwise be resampled label after label) are exactly the ones it removes, so the records
within a label are distinct receptors rather than one clone counted many times.

Input: dump/pairseq/subject{X,Y}/TCR{A,B}.well<NN>.tsv.gz -- Adaptive immunoSEQ well
exports, slimmed on the cluster to (aminoAcid, copy, vGeneName, jGeneName). Gene tokens keep
Adaptive's native naming (TCRBV07-09, TCRBJ02-03); no cross-dataset harmonisation is needed
because every metric is computed within a cohort against its own background.
See SOURCES.md for the fetch/regenerate command.

Usage: python src/build_pairseq.py
"""
from __future__ import annotations
import os, re, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import valid_cdr3, canon_gene

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRSEQ = os.path.join(REPO, "dump", "pairseq")
RESULTS = os.path.join(REPO, "results")

TOP = 50             # records per well per chain -- IMMREP25's 50 receptors per peptide
WELLS_PER_SUBJ = 10  # 10 wells x 2 subjects = 20 labels, as IMMREP25's 20 peptides
ENRICH = 10.0        # fold-enrichment in the well vs the mean of the other wells
SEED = 0
SUBJECTS = {"subjectX": "X", "subjectY": "Y"}


def _load_chain(subdir: str, chain: str) -> pd.DataFrame:
    """Long frame (well, cdr3, count, v, j) over every well of one subject/chain."""
    rows = []
    for p in sorted(glob.glob(os.path.join(PAIRSEQ, subdir, "TCR%s.well*.tsv.gz" % chain))):
        well = re.search(r"well(\d+)", os.path.basename(p)).group(1)
        d = pd.read_csv(p, sep="\t")
        d = d[d.aminoAcid.map(valid_cdr3).astype(bool)]
        if d.empty:
            continue                       # failed well (subjectX well25 is empty at source)
        rows.append(pd.DataFrame({"well": well, "cdr3": d.aminoAcid.to_numpy(),
                                  "count": d["copy"].to_numpy(),
                                  "v": [canon_gene(x) for x in d.vGeneName],
                                  "j": [canon_gene(x) for x in d.jGeneName]}))
    return pd.concat(rows, ignore_index=True)


def _enriched(long: pd.DataFrame, top: int = TOP, fold: float = ENRICH) -> dict:
    """Per well, the `top` clonotypes at least `fold`-fold enriched over the other wells.

    Frequencies are within-well (wells differ in sequencing depth). The comparison is against
    the mean frequency over *all* other wells, counting a well where the clonotype was not
    observed as zero -- so a clonotype seen in one well only is maximally enriched, which is
    precisely the call MIRA would make for it.
    """
    freq = long.assign(freq=long["count"] / long.groupby("well")["count"].transform("sum"))
    n_wells = freq.well.nunique()
    total = freq.groupby("cdr3")["freq"].transform("sum")
    # mean frequency across the OTHER wells; zero-observations included in the denominator
    others = (total - freq["freq"]) / max(n_wells - 1, 1)
    freq = freq.assign(enrich=freq["freq"] / others.where(others > 0, np.nan))
    freq["enrich"] = freq["enrich"].fillna(np.inf)      # unseen elsewhere -> fully well-specific
    out = {}
    for well, g in freq[freq.enrich >= fold].groupby("well"):
        out[well] = g.sort_values(["enrich", "freq"], ascending=False).head(top).reset_index(drop=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    rows, diag = [], []
    for subdir, label in SUBJECTS.items():
        ea = _enriched(_load_chain(subdir, "A"))
        eb = _enriched(_load_chain(subdir, "B"))
        # wells where both chains yield a full complement of enriched clonotypes
        usable = sorted(w for w in set(ea) & set(eb) if len(ea[w]) >= TOP and len(eb[w]) >= TOP)
        diag.append((label, len(ea), len(eb), len(usable)))
        for well in usable[:WELLS_PER_SUBJ]:
            a, b = ea[well], eb[well]
            # random alpha/beta combination WITHIN the well -- never across wells or subjects
            ia = rng.permutation(len(a))[:TOP]
            ib = rng.permutation(len(b))[:TOP]
            for i, j in zip(ia, ib):
                rows.append({"cohort": "pairseq_mock", "epitope": "%s_w%s" % (label, well),
                             "hla": label, "well": well,
                             "cdr3a": a.cdr3[i], "va": a.v[i], "ja": a.j[i],
                             "cdr3b": b.cdr3[j], "vb": b.v[j], "jb": b.j[j]})
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no usable pairSEQ wells under %s" % PAIRSEQ)
    out["tcr_id"] = ["ps_%d" % i for i in range(len(out))]
    out.to_csv(os.path.join(RESULTS, "pairseq_mock.tsv"), sep="\t", index=False)

    print("pairseq_mock: %d mock TCRs, %d labels (wells), %d HLA groups (subjects), "
          "%.0fx enrichment filter" % (len(out), out.epitope.nunique(), out.hla.nunique(), ENRICH))
    for label, na, nb, nu in diag:
        print("  subject %s: wells with enriched TCRA %d, TCRB %d, usable (>=%d both) %d"
              % (label, na, nb, TOP, nu))
    print("  distinct CDR3b %d/%d, CDR3a %d/%d"
          % (out.cdr3b.nunique(), len(out), out.cdr3a.nunique(), len(out)))


if __name__ == "__main__":
    main()
