"""Is "public" a proxy for "near-germline"? The Pgen-publicity association, measured here.
# 2026-11-05  (added at revision round 1 for reviewer 2.4, who noted that "public" is used
#              operationally -- near a receptor in a reference database -- rather than as a
#              claim about population frequency, and asked for the link to be made explicit)

The manuscript flags a positive as public when either chain lies within one substitution of a
VDJdb reference receptor, and then reads that as "shared, near-germline". The reviewer is right
that those are different statements: database proximity is not population frequency. What ties
them is generation probability. A receptor that recombination produces readily is generated
independently in many individuals, so it is both more likely to be shared between donors and more
likely to be in a reference database already -- the correspondence established across ~700 deeply
sequenced repertoires in Pogorelyy et al. (Genome Medicine 2018).

This module measures that correspondence on THIS dataset rather than citing it. For the IMMREP25
positives, each chain's OLGA generation probability is read off the committed per-chain cache and
compared against publicity, two ways:

  - as a correlation, Spearman rho between log10 Pgen and the number of distinct reference
    epitopes the chain's neighbours carry (a graded publicity degree, not a binary flag);
  - as an association across strata -- degree 0, 1, 2, 3+ -- by Kruskal-Wallis, which assumes
    nothing about the shape of the relationship and is the right test for an ordered grouping
    with a skewed response.

Both are reported per chain. The claim being supported is modest and should stay so: publicity as
operationalised here tracks generation probability, which is why the two words are used close
together. It is not a measurement of how often these receptors occur in people.

Usage: uv run python src/pgen_publicity.py          # writes results/ + macros
       uv run python src/pgen_publicity.py --demo   # self-checks only, no writes
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import build_cohorts                                   # noqa: E402
from load_data import load_vdjdb                                    # noqa: E402
from crossreact_1mm import epitope_index, neighbour_epitopes        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")

STRATA = ("0", "1", "2", "3+")


def degree_stratum(d: int) -> str:
    """Publicity degree bucketed as the reviewer framed it: found 0, 1, 2 or 3+ times."""
    return "3+" if d >= 3 else str(d)


def fmt_p(p: float) -> str:
    """LaTeX-ready p-value: an explicit bound below 1e-4, else two significant figures."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "$P<10^{-4}$"
    return "$P=%.2g$" % p


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)
    coh = build_cohorts()
    imm = coh["immrep25_pos"]["paired"]
    vlong, _ = load_vdjdb()

    rows = []
    macros: dict[str, str] = {}
    for chain, col in (("A", "cdr3a"), ("B", "cdr3b")):
        idx = epitope_index(vlong, chain)
        deg = np.array([len(neighbour_epitopes(s, idx)) for s in imm[col]], float)
        pg = (pd.read_csv(os.path.join(REPO, "cache", "pgen1mm_immrep25_pos_%s.tsv" % chain),
                          sep="\t")
              .drop_duplicates("cdr3").set_index("cdr3").pgen)
        p = imm[col].map(pg).to_numpy(float)
        ok = np.isfinite(p) & (p > 0)
        lp = np.log10(p[ok])
        dg = deg[ok]

        rho, p_rho = spearmanr(lp, dg)
        groups = [lp[[degree_stratum(int(d)) == s for d in dg]] for s in STRATA]
        present = [g for g in groups if len(g) >= 2]
        H, p_kw = kruskal(*present) if len(present) >= 2 else (np.nan, np.nan)

        for s, g in zip(STRATA, groups):
            rows.append({"chain": chain, "stratum": s, "n": len(g),
                         "median_log10_pgen": float(np.median(g)) if len(g) else float("nan")})

        meds = {s: (float(np.median(g)) if len(g) else float("nan"))
                for s, g in zip(STRATA, groups)}
        macros.update({
            "pp%sRho" % chain: "%.3f" % rho,
            "pp%sRhoP" % chain: fmt_p(p_rho),
            "pp%sH" % chain: "%.1f" % H,
            "pp%sKwP" % chain: fmt_p(p_kw),
            "pp%sN" % chain: "{:,}".format(int(ok.sum())).replace(",", "\\,"),
            "pp%sMedZero" % chain: "%.1f" % meds["0"],
            "pp%sMedThree" % chain: "%.1f" % meds["3+"],
            "pp%sGap" % chain: "%.1f" % (meds["3+"] - meds["0"]),
            "pp%sNzero" % chain: "{:,}".format(len(groups[0])).replace(",", "\\,"),
            "pp%sNthree" % chain: "{:,}".format(len(groups[3])).replace(",", "\\,"),
        })
        print("TCR%s  n=%d  Spearman rho(log10 Pgen, publicity degree) = %.3f (%s); "
              "Kruskal-Wallis across degree 0/1/2/3+ H=%.1f (%s); median log10 Pgen "
              "%.1f at degree 0 against %.1f at 3+"
              % (chain, int(ok.sum()), rho, fmt_p(p_rho), H, fmt_p(p_kw),
                 meds["0"], meds["3+"]))

    with open(os.path.join(RESULTS, "pgen_publicity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["chain", "stratum", "n", "median_log10_pgen"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(ADAT, "pgen_publicity_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))


def demo():
    assert degree_stratum(0) == "0" and degree_stratum(2) == "2"
    assert degree_stratum(3) == "3+" and degree_stratum(17) == "3+"
    assert fmt_p(1e-9).startswith("$P<") and "0.03" in fmt_p(0.031)

    # a planted monotone association must be recovered with the expected sign, and a null one not
    rng = np.random.default_rng(0)
    lp = rng.normal(-9, 1.5, 600)
    deg = np.clip(np.round(2.5 * (lp + 9) / 1.5 + rng.normal(0, 1, 600)), 0, None)
    rho, p = spearmanr(lp, deg)
    assert rho > 0.5 and p < 1e-10, (rho, p)
    rho0, p0 = spearmanr(lp, rng.permutation(deg))
    assert abs(rho0) < 0.15, rho0

    # Kruskal-Wallis must see a shift between strata and not see one when there is none
    g = [rng.normal(m, 1.0, 120) for m in (-10.0, -9.0, -8.0, -7.0)]
    assert kruskal(*g).pvalue < 1e-10
    flat = [rng.normal(-9.0, 1.0, 120) for _ in range(4)]
    assert kruskal(*flat).pvalue > 0.01
    print("demo OK: stratum bucketing, p formatting, and both tests behave on planted and "
          "null associations")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
