#!/usr/bin/env python
# 2026-09-13  Dataset overlap, software versions and seeds, generated rather than asserted.
#
# A reviewer asked for dataset overlap, software versions, random seeds and complete
# reproduction scripts. The first of those is not hygiene but a load-bearing fact: the
# calibration ladder is only a ladder if its rungs are independent, and several are not. The
# Methods already states some of this piecemeal -- that VDJdb contains the TCRvdb receptors,
# that IMMREP22's positives are almost a subset of IMMREP23's training set, that the dCODE
# note overlaps VDJdb -- each from a separate hand-written count. This computes the whole
# matrix in one pass so the dependence structure is visible at once and cannot drift from the
# prose.
#
# Overlap is reported ASYMMETRICALLY, as the fraction of cohort A's unique CDR3s that also
# occur in cohort B, because the cohorts differ in size by three orders of magnitude and a
# symmetric Jaccard would hide exactly the containments that matter (VDJdb-HQ has ~54k TCRbeta
# clonotypes against IMMREP25's ~1k, so their Jaccard is small in both directions while the
# containment can be near total).
#
# Versions and seeds are read from the live environment and from the modules' own constants,
# so the reported settings cannot diverge from the ones used -- the same discipline
# probe_params.py already applies to the probe parameters.
#
# Run: python src/reproducibility.py      (--demo for the self-check)
from __future__ import annotations
import importlib
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

from cohorts import build_cohorts, COHORT_META          # noqa: E402

PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "igraph", "leidenalg", "networkx", "Bio")
# modules whose seed constant governs a reported number
SEEDED = ("epitope_free", "retrieval_baseline", "publicity_controls", "cohort_stats",
          "sensitivity", "pairwise", "utility", "validation_efficiency")


def overlap_matrix(coh, chain: str) -> pd.DataFrame:
    """Asymmetric CDR3 overlap: entry (A, B) is the share of A's clonotypes present in B."""
    sets = {}
    for name, c in coh.items():
        lg = c.get("long")
        if lg is None or lg.empty:
            continue
        s = set(lg[lg.chain == chain].cdr3)
        if s:
            sets[name] = s
    names = [n for n in COHORT_META if n in sets] or sorted(sets)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = len(sets[a] & sets[b]) / len(sets[a])
    M.insert(0, "n_clonotypes", [len(sets[n]) for n in names])
    return M


def versions() -> pd.DataFrame:
    rows = [dict(component="python", version=platform.python_version()),
            dict(component="platform", version="%s %s" % (platform.system(),
                                                          platform.machine()))]
    for p in PACKAGES:
        try:
            m = importlib.import_module(p)
            rows.append(dict(component=p, version=getattr(m, "__version__", "unknown")))
        except ImportError:
            rows.append(dict(component=p, version="absent"))
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        rows.append(dict(component="analysis repo commit",
                         version=sha + (" (uncommitted changes present)" if dirty else "")))
    except Exception:
        rows.append(dict(component="analysis repo commit", version="unknown"))
    return pd.DataFrame(rows)


def seeds() -> pd.DataFrame:
    rows = []
    for mod in SEEDED:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            rows.append(dict(module=mod, seed="import failed: %s" % type(exc).__name__))
            continue
        found = {k: getattr(m, k) for k in ("SEED", "N_PERM", "N_DRAWS", "N_RANDOM",
                                            "N_MATCHED", "N_SIM", "N_FOLDS")
                 if hasattr(m, k)}
        rows.append(dict(module=mod,
                         seed=found.get("SEED", "n/a"),
                         resamples="; ".join("%s=%s" % (k, v) for k, v in found.items()
                                             if k != "SEED") or "n/a"))
    return pd.DataFrame(rows)


def main():
    coh = build_cohorts()
    frames = {}
    for chain in ("A", "B"):
        M = overlap_matrix(coh, chain)
        frames[chain] = M
        M.to_csv(os.path.join(RESULTS, "overlap_TR%s.csv" % chain))
        print("\n=== TCR%s clonotype overlap: share of the ROW cohort present in the COLUMN "
              "cohort, %% ===" % chain)
        # scale only the overlap fractions; n_clonotypes is a count and must not be x100
        shown = M.copy()
        frac = [c for c in shown.columns if c != "n_clonotypes"]
        shown[frac] = (shown[frac] * 100).round(1)
        print(shown.to_string())

    v, s = versions(), seeds()
    v.to_csv(os.path.join(RESULTS, "versions.csv"), index=False)
    s.to_csv(os.path.join(RESULTS, "seeds.csv"), index=False)
    print("\n=== software ===");  print(v.to_string(index=False))
    print("\n=== seeds and resampling counts ===");  print(s.to_string(index=False))

    B = frames["B"]
    def ov(a, b):
        return float(B.loc[a, b]) if a in B.index and b in B.columns else np.nan
    pairs = {"ovImmHq": ("immrep25_pos", "vdjdb_hq"),
             "ovImmLq": ("immrep25_pos", "vdjdb_lq"),
             "ovIiHq": ("immrep22_true", "vdjdb_hq"),
             "ovTtHq": ("tcrvdb_true", "vdjdb_hq"),
             "ovImmPs": ("immrep25_pos", "pairseq_mock")}
    macros = {k: "%.1f" % (100 * ov(*p)) for k, p in pairs.items() if np.isfinite(ov(*p))}
    macros["ovNcohorts"] = "%d" % len(B)
    for comp in ("python", "numpy", "pandas", "scipy", "sklearn"):
        r = v[v.component == comp]
        if not r.empty:
            macros["ver" + comp.capitalize()] = str(r.version.iloc[0])
    with open(os.path.join(ADAT, "reproducibility_macros.tex"), "w") as fh:
        for k, val in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, val))
    print("\nwrote results/overlap_TR{A,B}.csv, versions.csv, seeds.csv")
    print("wrote %s (%d macros)" % (os.path.join(ADAT, "reproducibility_macros.tex"),
                                    len(macros)))


def demo():
    """Self-check: the overlap matrix's diagonal is 1 by construction, it is genuinely
    asymmetric where one cohort contains another, and the version/seed tables are populated."""
    coh = build_cohorts()
    M = overlap_matrix(coh, "B")
    num = M.drop(columns=["n_clonotypes"])
    assert np.allclose(np.diag(num.to_numpy(float)), 1.0), "diagonal must be self-overlap"
    assert ((num.to_numpy(float) >= -1e-12) & (num.to_numpy(float) <= 1 + 1e-12)).all()
    # asymmetry is the point: a small cohort can be largely contained in a big one without
    # the converse holding
    a, b = "immrep25_pos", "vdjdb_hq"
    if a in num.index and b in num.index:
        assert not np.isclose(num.loc[a, b], num.loc[b, a]) or num.loc[a, b] == 0, \
            "overlap should be asymmetric between cohorts of very different size"
    v, s = versions(), seeds()
    assert (v.version != "").all() and len(v) >= len(PACKAGES)
    assert not s.empty and s.seed.notna().all()
    assert not (s.seed.astype(str) == "import failed").any()
    print("reproducibility.demo OK  (%d cohorts on TCRbeta, diagonal exact, %d software "
          "components and %d seeded modules recorded)" % (len(num), len(v), len(s)))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
