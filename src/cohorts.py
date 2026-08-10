"""Assemble the six audit cohorts into the common long/paired schemas.

Each cohort is a dict {'long': <per-chain df>, 'paired': <wide df or None>}.
Cohort order encodes the *expected* signal-to-noise hierarchy (high -> low).
"""
from __future__ import annotations
import os
import pandas as pd

from load_data import (load_immrep, load_immrep22, load_tcrvdb, load_vdjdb,
                       _pair_to_long, valid_cdr3, LONG_COLS, PAIR_COLS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kept-in-bars cohorts occupy 0-7; the two controls dropped from the bar/ESM panels
# (airr_control=AIRR random, olga_random=OLGA) sit last (8,9) so bar xranges truncate them
# cleanly. They remain referenced by name in the pgen figures + homology graphs.
# Primary cohorts (0-6) are the benchmark-like sets scored on their own labels; the
# auxiliary nulls (7-9) fix the floor and are shown only where a floor is needed.
HIERARCHY = ["tcrvdb_true", "vdjdb_hq", "immrep22_true", "vdjdb_lq", "tcrvdb_false",
             "immrep25_pos", "pairseq_mock",
             "mlr_prolif", "airr_control", "olga_random"]

# nice labels + a stable colour index for figures
COHORT_META = {
# 'label' is the plain form used in results/*.csv; 'tex' is the typeset form used in every
# figure and table. A parenthesised superscript is reserved for the subset of a resource a
# cohort is drawn from -- (+)/(-) its positive/negative cases, (HQ)/(LQ) its reference-support
# tiers; the controls take plain-word suffixes instead.
    "tcrvdb_true":  {"label": "TCRvdb pos",  "tex": r"TCRvdb$^{(+)}$",    "expect": "signal"},
    "vdjdb_hq":     {"label": "VDJdb HQ",    "tex": r"VDJdb$^{(HQ)}$",    "expect": "signal"},
    "immrep22_true": {"label": "IMMREP22 pos", "tex": r"IMMREP22$^{(+)}$", "expect": "signal"},
    "vdjdb_lq":     {"label": "VDJdb LQ",    "tex": r"VDJdb$^{(LQ)}$",    "expect": "weak"},
    "tcrvdb_false": {"label": "TCRvdb neg",  "tex": r"TCRvdb$^{(-)}$",    "expect": "noise"},
    "immrep25_pos": {"label": "IMMREP25 pos", "tex": r"IMMREP25$^{(+)}$", "expect": "test"},
    "airr_control": {"label": "AIRR rand",   "tex": r"AIRR random",   "expect": "real"},
    "pairseq_mock": {"label": "pairSEQ mock", "tex": r"pairSEQ mock", "expect": "test"},
    "olga_random":  {"label": "OLGA rand",   "tex": r"OLGA random",   "expect": "noise"},
    "mlr_prolif":   {"label": "MLR expanded", "tex": r"MLR expanded",     "expect": "real"},
}


def _mk_paired(df: pd.DataFrame, cohort: str, quality: str) -> pd.DataFrame:
    """Normalise a paired frame and reduce it to distinct clonotypes per epitope.

    De-duplication is applied to every cohort, not only the ones that aggregate studies, so
    that the ladder compares like with like: a repeated record is a statement about how often
    a receptor was reported, not about how convergent an epitope's repertoire is, and the pair
    exposures of any similarity statistic scale with the square of it.
    """
    out = df.copy()
    out["cohort"] = cohort
    out["quality"] = quality
    out = out[out["cdr3a"].map(valid_cdr3) & out["cdr3b"].map(valid_cdr3)]
    out = out.drop_duplicates(["epitope", "cdr3a", "cdr3b"])
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

    # --- immrep22 true positives (VDJdb-derived "seen" benchmark) ---
    im22 = load_immrep22()
    p22 = _mk_paired(im22, "immrep22_true", "true")
    coh["immrep22_true"] = {"paired": p22, "long": _pair_to_long(p22)}

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
        for name in ("airr_control", "olga_random", "pairseq_mock"):
            src = _load_olga(name)
            if src is None:
                continue
            p = _mk_paired(src, name, "noise")
            coh[name] = {"paired": p, "long": _pair_to_long(p)}

        # beta-only expanded-clone control (already in long form, carries `reaction`)
        mlr = _load_olga("mlr_prolif")
        if mlr is not None:
            lg = mlr[mlr.cdr3.map(valid_cdr3)]
            lg = lg.drop_duplicates(["epitope", "cdr3"]).reset_index(drop=True)
            coh["mlr_prolif"] = {"paired": None, "long": lg}

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
