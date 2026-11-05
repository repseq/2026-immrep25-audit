"""Does interface confidence discriminate at all? The external check, on a validated benchmark.
# 2026-11-05  (added at revision round 1 for reviewers 2 (minor, ipTM) and 3.4)

Two reviewers read the same observation in opposite directions, and both readings need answering.
The manuscript reports that TCRmodel2 models the IMMREP25 complexes with a median ipTM close to
that of validated binders and of mismatched decoys alike. One reviewer concluded that ipTM is
therefore not a useful discriminator at all. The other warned that treating a drop in ipTM as
absence of binding confuses docking plausibility with binding free energy, and that this should
not be read as ruling out physics-based rescoring in general.

Neither conclusion follows from the medians, and the evidence that settles it is external to this
study. On the functionally validated TCRvdb/IMMREP26 benchmark -- real curated pairings that an
independent screen either did or did not reproduce -- interface confidence separates the two
classes WITHIN an epitope at a useful rate. So ipTM is not blind; what it cannot do is compare
across epitopes, where template coverage and peptide chemistry shift the whole distribution. That
is a calibration limit, not an absence of signal, and it is exactly why the manuscript's
structural argument rests on chance-corrected within-epitope indices rather than on raw ipTM.

This module only re-reads a result table produced elsewhere (the IMMREP26 ranking work) and emits
summary statistics. It fits nothing and recomputes nothing, so it cannot perturb a standing value,
and it writes no per-receptor data: TCRvdb is private and only derived statistics may leave it.

Usage: uv run python src/iptm_external.py          # writes results/ + macros
       uv run python src/iptm_external.py --demo   # self-checks only, no writes
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAT = os.path.join(REPO, "appendix", "analysis")
RESULTS = os.path.join(REPO, "results")
SRC = os.path.join(REPO, "dump", "immrep26", "immrep26_public_metrics.csv")

# The two confidence-only scorers of that benchmark: raw interface ipTM, and ipTM combined with a
# topological summary of the interface. `tcren_full` is the full statistical-potential scorer and
# is NOT a confidence metric, so it is reported only as the context in which the others sit.
CONF = ("tcren_iptm", "tcren_iptm_betti")
CONTEXT = "tcren_full"
MACRO = "macro"


def load(path: str = SRC) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {"method", "epitope_aa", "records", "positive", "negative", "roc_auc"}
    assert need <= set(d.columns), sorted(need - set(d.columns))
    d["roc_auc"] = pd.to_numeric(d.roc_auc, errors="coerce")
    return d


def arm(d: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-epitope rows for one method, with the pooled macro row separated out."""
    s = d[d.method == method]
    return s[s.epitope_aa != MACRO].sort_values("epitope_aa"), s[s.epitope_aa == MACRO]


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ADAT, exist_ok=True)
    d = load()

    rows = []
    macros: dict[str, str] = {}
    for method in CONF + (CONTEXT,):
        per, mac = arm(d, method)
        assert not per.empty, "no per-epitope rows for %s" % method
        assert len(mac) == 1, "expected exactly one macro row for %s" % method
        for _, r in per.iterrows():
            rows.append({"method": method, "epitope": r.epitope_aa, "n": int(r.records),
                         "positive": int(r.positive), "negative": int(r.negative),
                         "roc_auc": float(r.roc_auc)})
        key = {"tcren_iptm": "Iptm", "tcren_iptm_betti": "Betti",
               CONTEXT: "Full"}[method]
        macros.update({
            "ix%sMacro" % key: "%.3f" % float(mac.roc_auc.iloc[0]),
            "ix%sLo" % key: "%.3f" % float(per.roc_auc.min()),
            "ix%sHi" % key: "%.3f" % float(per.roc_auc.max()),
        })

    per_iptm, mac_iptm = arm(d, "tcren_iptm")
    macros.update({
        "ixNepi": "%d" % len(per_iptm),
        "ixNrec": "{:,}".format(int(mac_iptm.records.iloc[0])).replace(",", "\\,"),
        "ixNpos": "{:,}".format(int(mac_iptm.positive.iloc[0])).replace(",", "\\,"),
        "ixNneg": "{:,}".format(int(mac_iptm.negative.iloc[0])).replace(",", "\\,"),
        "ixWeakPep": per_iptm.loc[per_iptm.roc_auc.idxmin(), "epitope_aa"],
        "ixStrongPep": per_iptm.loc[per_iptm.roc_auc.idxmax(), "epitope_aa"],
    })

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "iptm_external.csv"), index=False)
    with open(os.path.join(ADAT, "iptm_external_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    print("on the functionally validated benchmark (%d epitopes, %d records: %d reproduced, "
          "%d not), interface confidence alone separates the two classes within an epitope at "
          "macro ROC AUC %.3f (%.3f on %s to %.3f on %s)"
          % (len(per_iptm), int(mac_iptm.records.iloc[0]), int(mac_iptm.positive.iloc[0]),
             int(mac_iptm.negative.iloc[0]), float(mac_iptm.roc_auc.iloc[0]),
             per_iptm.roc_auc.min(), macros["ixWeakPep"], per_iptm.roc_auc.max(),
             macros["ixStrongPep"]))
    print("so a flat median across cohorts is a calibration limit across epitopes, not an "
          "absence of signal within one")


def demo():
    d = load()
    per, mac = arm(d, "tcren_iptm")
    # the table must be the public recalibrated endpoint, 5 epitopes plus a macro row
    assert len(per) == 5 and len(mac) == 1, (len(per), len(mac))
    # every AUC must be a probability, and the macro must lie within the per-epitope range
    assert per.roc_auc.between(0, 1).all()
    assert per.roc_auc.min() <= float(mac.roc_auc.iloc[0]) <= per.roc_auc.max()
    # the claim being made: confidence alone is well above chance on this benchmark
    assert float(mac.roc_auc.iloc[0]) > 0.70, float(mac.roc_auc.iloc[0])
    assert per.roc_auc.min() > 0.5, "an epitope at or below chance would weaken the claim"
    # class counts must add up
    assert (per.positive + per.negative == per.records).all()
    print("demo OK: %d epitopes, macro AUC %.3f, per-epitope %.3f-%.3f, class counts consistent"
          % (len(per), float(mac.roc_auc.iloc[0]), per.roc_auc.min(), per.roc_auc.max()))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
