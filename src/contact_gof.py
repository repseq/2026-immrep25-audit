#!/usr/bin/env python
# 2026-08-06  Do the IMMREP25 predicted TCR-peptide interfaces reproduce the residue-contact
# STATISTICS of real TCR-pMHC binding? -- via the TCRen statistical-potential approach.
#
# TCRen is a knowledge-based contact potential: a per-residue-pair log-odds  log(observed/expected)
# derived from real crystal contacts. We apply the same construction to the *predicted* contacts of a
# cohort (its own statistical potential) and ask two questions against the real TCRen potential:
#   (R)  does the cohort's contact potential correlate with the real one?  (structure-level bootstrap CI)
#   (S)  are the cohort's contacts favourable under the real potential?    (mean per-contact log-odds)
# A real binder run through the same TCRmodel2 pipeline (vdjdb_binder) is the positive control: it
# should recover the real potential. If IMMREP25 does not, its folded interfaces do not carry genuine
# binding contact chemistry -- consistent with the positives not being cognate pairs.
#
# NB we compare on the ODDS (enrichment) scale, not raw contact frequencies: a frequency correlation
# is dominated by amino-acid composition (abundant residues contact often in any interface) and is
# therefore uninformative about binding-specific contact preferences.
#
# Contacts are pre-extracted with tcren's batch command (run in tcren's env):
#   ~/vcs/code/tcren/.venv/bin/tcren contacts -s cache/struct/<cohort>/ \
#       -o cache/struct/contacts_<cohort>.csv --interface tcr_peptide --regions all
# This script runs in the audit env (pandas/numpy/scipy):  python src/contact_gof.py
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
COHORTS = {  # label -> pre-extracted contact table
    "immrep25": REPO / "cache/struct/contacts_immrep25.csv",
    "vdjdb_binder": REPO / "cache/struct/contacts_vdjdb_binder.csv",
}
TCREN_CSV = Path.home() / "vcs/code/tcren/src/tcren/data/TCRen_potential.csv"
OUT_CSV = REPO / "results" / "contact_gof.csv"
MACROS = REPO / "appendix" / "analysis" / "contact_gof_macros.tex"
NBOOT = 1000
SEED = 0

AA = set("ACDEFGHIKLMNPQRSTVWY")
F, T = "residue.aa.from", "residue.aa.to"


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[d[F].isin(AA) & d[T].isin(AA)].dropna(subset=[F, T])


def potential(df: pd.DataFrame) -> dict:
    """TCRen-style statistical potential from observed contacts: log(obs/exp), exp = marginal product."""
    c = df.groupby([F, T]).size()
    tot = c.sum()
    pf = c.groupby(level=0).sum() / tot
    pt = c.groupby(level=1).sum() / tot
    return {k: float(np.log((c[k] / tot) / (pf[k[0]] * pt[k[1]]))) for k in c.index}


def analyse(df: pd.DataFrame, tcren: dict, rng) -> dict:
    E = potential(df)
    keys = [k for k in E if k in tcren]
    a = np.array([E[k] for k in keys])
    b = np.array([-tcren[k] for k in keys])            # -TCRen = real log-odds (enrichment)
    R, pR = stats.pearsonr(a, b)

    # structure-level bootstrap: resample complexes, re-derive the cohort potential, re-correlate
    ids = df["pdb.id"].unique()
    groups = {i: g for i, g in df.groupby("pdb.id")}
    Rs = np.empty(NBOOT)
    for j in range(NBOOT):
        s = pd.concat([groups[i] for i in rng.choice(ids, len(ids), replace=True)])
        Eb = potential(s)
        kk = [k for k in Eb if k in tcren]
        Rs[j] = stats.pearsonr([Eb[k] for k in kk], [-tcren[k] for k in kk])[0]
    lo, hi = np.percentile(Rs, [2.5, 97.5])

    # mean per-contact real-potential score (favourable-enrichment test, H0: 0 = composition null)
    score = np.array([-tcren[(x, y)] for x, y in zip(df[F], df[T]) if (x, y) in tcren])
    tstat, pS = stats.ttest_1samp(score, 0)
    return dict(n_struct=int(df["pdb.id"].nunique()), n_contacts=int(len(df)), n_pairs=len(keys),
                R=R, R_p=pR, R_lo=lo, R_hi=hi, R_frac_le0=float(np.mean(Rs <= 0)),
                mean_score=float(score.mean()), score_t=float(tstat), score_p=float(pS))


def main() -> None:
    pot = pd.read_csv(TCREN_CSV)
    tcren = {(r[F], r[T]): float(r["TCRen"]) for _, r in pot.iterrows()}
    rng = np.random.default_rng(SEED)

    res = {name: analyse(load(p), tcren, rng) for name, p in COHORTS.items() if p.exists()}
    rows = [{"cohort": n, **v} for n, v in res.items()]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    imm, bind = res["immrep25"], res.get("vdjdb_binder")
    sig = lambda p: "<10^{-4}" if p < 1e-4 else f"={p:.3f}"
    macros = [
        (r"\gofNstruct", f"{imm['n_struct']}"),
        (r"\gofNcontacts", f"{imm['n_contacts']:,}".replace(",", "{,}")),
        (r"\gofRimm", f"{imm['R']:.2f}"),
        (r"\gofRimmLo", f"{imm['R_lo']:.2f}"), (r"\gofRimmHi", f"{imm['R_hi']:.2f}"),
        (r"\gofScoreImm", f"{imm['mean_score']:+.2f}"),
        (r"\gofScoreImmP", sig(imm["score_p"])),
    ]
    if bind:
        macros += [
            (r"\gofNbind", f"{bind['n_struct']}"),
            (r"\gofRbind", f"{bind['R']:.2f}"),
            (r"\gofRbindLo", f"{bind['R_lo']:.2f}"), (r"\gofRbindHi", f"{bind['R_hi']:.2f}"),
            (r"\gofScoreBind", f"{bind['mean_score']:+.2f}"),
        ]
    MACROS.write_text("".join(f"\\newcommand{{{k}}}{{{v}}}\n" for k, v in macros))

    print("=== TCRen statistical-potential agreement (predicted vs real crystals) ===")
    for n, v in res.items():
        print(f"  {n:14s} n={v['n_struct']:4d} contacts={v['n_contacts']:6d}  "
              f"R={v['R']:+.3f} [95% CI {v['R_lo']:+.3f},{v['R_hi']:+.3f}] frac<=0={v['R_frac_le0']:.3f}  "
              f"mean-score={v['mean_score']:+.3f} nats (t={v['score_t']:.1f}, p={v['score_p']:.2g})")
    print(f"\nwrote {OUT_CSV}\nwrote {MACROS}")


if __name__ == "__main__":
    main()
