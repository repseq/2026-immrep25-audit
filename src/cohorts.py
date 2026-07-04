"""Assemble the six audit cohorts into the common long/paired schemas.

Each cohort is a dict {'long': <per-chain df>, 'paired': <wide df or None>}.
Cohort order encodes the *expected* signal-to-noise hierarchy (high -> low).
"""
from __future__ import annotations
import os
import pandas as pd

from load_data import (load_immrep, load_tcrvdb, load_vdjdb,
                       _pair_to_long, valid_cdr3, LONG_COLS, PAIR_COLS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIERARCHY = ["tcrvdb_true", "vdjdb_hq", "vdjdb_lq", "tcrvdb_false",
             "immrep25_pos", "airr_control", "olga_matched", "olga_random"]

# nice labels + a stable colour index for figures
COHORT_META = {
    "tcrvdb_true":  {"label": "TCRvdb true",  "expect": "signal"},
    "vdjdb_hq":     {"label": "VDJdb HQ",     "expect": "signal"},
    "vdjdb_lq":     {"label": "VDJdb LQ",     "expect": "weak"},
    "tcrvdb_false": {"label": "TCRvdb false", "expect": "noise"},
    "immrep25_pos": {"label": "immrep25 pos", "expect": "test"},
    "airr_control": {"label": "AIRR control", "expect": "real"},
    "olga_matched": {"label": "OLGA pgen-matched", "expect": "noise"},
    "olga_random":  {"label": "OLGA random",  "expect": "noise"},
}


def _mk_paired(df: pd.DataFrame, cohort: str, quality: str) -> pd.DataFrame:
    out = df.copy()
    out["cohort"] = cohort
    out["quality"] = quality
    out = out[out["cdr3a"].map(valid_cdr3) & out["cdr3b"].map(valid_cdr3)]
    return out[PAIR_COLS].reset_index(drop=True)


def _load_olga(name: str):
    p = os.path.join(_REPO, "results", "%s.tsv" % name)
    return pd.read_csv(p, sep="\t") if os.path.exists(p) else None


def build_cohorts(include_olga: bool = False, olga_paired: pd.DataFrame | None = None) -> dict:
    coh: dict[str, dict] = {}

    # --- immrep25 positives ---
    im = load_immrep()
    pos = im[im.label == 1]
    p = _mk_paired(pos, "immrep25_pos", "pos")
    coh["immrep25_pos"] = {"paired": p, "long": _pair_to_long(p)}

    # --- TCRvdb true / false ---
    tv = load_tcrvdb()
    for name, mask in [("tcrvdb_true", tv.padj < 1e-5),
                       ("tcrvdb_false", tv.padj >= 1e-5)]:
        p = _mk_paired(tv[mask], name, name.split("_")[1])
        coh[name] = {"paired": p, "long": _pair_to_long(p)}

    # --- VDJdb HQ / LQ ---
    long_all, paired = load_vdjdb()
    for name, q in [("vdjdb_hq", "hq"), ("vdjdb_lq", "lq")]:
        lg = long_all[long_all.quality == q].assign(cohort=name)[LONG_COLS].reset_index(drop=True)
        pr = paired[paired.quality == q].assign(cohort=name)[PAIR_COLS].reset_index(drop=True)
        coh[name] = {"paired": pr, "long": lg}

    # --- generated / real controls (results/<name>.tsv from build_olga.py, build_airr.py) ---
    if include_olga:
        for name in ("airr_control", "olga_matched", "olga_random"):
            src = _load_olga(name)
            if src is None:
                continue
            p = _mk_paired(src, name, "noise")
            coh[name] = {"paired": p, "long": _pair_to_long(p)}

    return coh


def counts_table(coh: dict) -> pd.DataFrame:
    rows = []
    for name in HIERARCHY:
        if name not in coh:
            continue
        lg, pr = coh[name]["long"], coh[name]["paired"]
        rows.append({
            "cohort": name,
            "label": COHORT_META[name]["label"],
            "n_paired": 0 if pr is None else len(pr),
            "n_chainA": int((lg.chain == "A").sum()),
            "n_chainB": int((lg.chain == "B").sum()),
            "n_epitopes": lg.epitope.nunique(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    coh = build_cohorts()
    with pd.option_context("display.width", 200):
        print(counts_table(coh).to_string(index=False))
