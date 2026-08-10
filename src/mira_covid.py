"""Does the MIRA assay itself produce convergent positive sets? Test on ImmuneCODE MIRA-COVID.
# 2026-08-06

IMMREP25's TCR->peptide labels come from a MIRA assay (peptide-pool expansion + pooled read-out).
A natural control is a DIFFERENT MIRA dataset: the ImmuneCODE MIRA-COVID release (Nolan/Adaptive
2020, class-I `peptide-detail-ci.csv`), which assigns TCRbeta to SARS-CoV-2 peptide pools by the
same assay. We run the audit homology probe on it (beta, grouped by pool). If MIRA-COVID shows CDR3
convergence, the MIRA assay can produce convergent sets, so IMMREP25's floor reflects its (unseen,
weakly-immunogenic) peptide selection rather than the assay; if it also floors, the assay itself is
implicated.

Fetch: the release zip is public (see SOURCES); this reads the extracted
dump/immunecode_mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv.
Usage: uv run python src/mira_covid.py   # writes results/mira_covid.csv + macros
"""
from __future__ import annotations
import os, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from homology import per_epitope_sn, dataset_sn                # noqa: E402
from load_data import valid_cdr3                               # noqa: E402

RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
CI = os.path.join(REPO, "dump", "immunecode_mira",
                  "ImmuneCODE-MIRA-Release002.1", "peptide-detail-ci.csv")
MIN_N = 30
CAP = 300         # cap TCRs/pool (bounds the O(n^2) homology count; matches audit practice)


def run():
    d = pd.read_csv(CI, usecols=["TCR BioIdentity", "Amino Acids"])
    d["cdr3"] = d["TCR BioIdentity"].str.split("+").str[0]
    d = d[d.cdr3.map(valid_cdr3)]
    d = d.rename(columns={"Amino Acids": "epitope"}).drop_duplicates(["epitope", "cdr3"])
    vc = d.epitope.value_counts()
    pools = vc[vc >= MIN_N].index
    d = d[d.epitope.isin(pools)]
    # cap oversized pools so the pairwise counting stays bounded (shuffle then take CAP/pool)
    d = d.sample(frac=1, random_state=0).groupby("epitope").head(CAP).reset_index(drop=True)
    lc = d[["epitope", "cdr3"]].assign(chain="B")

    pe = per_epitope_sn(lc, max_d=3, min_n=MIN_N, seed=1)
    snb = dataset_sn(pe, 1)["sn"]
    pe = pe.sort_values("sn1", ascending=False)
    pe[["epitope", "n", "sn1"]].to_csv(os.path.join(RESULTS, "mira_covid.csv"), index=False)
    npool = pe.epitope.nunique(); nabove = int((pe.sn1 > 2).sum())
    print("MIRA-COVID (class I) homology S/N beta = %.2f over %d pools; %d/%d pools S/N>2"
          % (snb, npool, nabove, npool))
    print(pe[["epitope", "n", "sn1"]].head(8).to_string(index=False))

    macros = {"miraHomB": "%.1f" % snb, "miraNpool": "%d" % npool,
              "miraNabove": "%d" % nabove, "miraMaxSnb": "%.0f" % pe.sn1.max()}
    with open(os.path.join(ADAT, "mira_covid_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("wrote results/mira_covid.csv + mira_covid_macros.tex")


if __name__ == "__main__":
    run()
