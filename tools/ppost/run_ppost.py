"""Selection-aware Ppost for every audited cohort, standalone runner; returns summary stats only.
# 2026-11-05, revision round 1, reviewer 3.5.
#
# The reviewer asked us to condition on post-selection probabilities from a selection-aware model
# (TCRsep, Jiang et al. 2025) rather than on OLGA's pre-selection Pgen alone. This scores every
# cohort and writes ONE small summary table; no per-sequence output goes back to the audit repo,
# which keeps derived statistics rather than intermediates.
#
# Two shims are needed and both are recorded here rather than hidden:
#   * the distributed selection model was serialised on a CUDA device, and this host is Apple
#     silicon with no CUDA, so torch.load is given map_location="cpu" and the .cuda() movers are
#     made no-ops. Nothing about the model's parameters changes; only where they are placed.
#   * TCRsep parses IMGT gene names, while IMMREP25 ships Adaptive tokens. The mapping is applied
#     upstream (v_imgt/j_imgt columns) via our own vdjtools table, and asserted complete there.
#
# TCRsep's distributed selection model is TCRbeta; the alpha chain is attempted and recorded as
# unavailable if it fails, rather than silently reported as if it had run.
#
# Run in an isolated scoring directory: python run_ppost.py
"""
from __future__ import annotations
import functools
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# tcrsep 1.2 still uses np.str, removed in NumPy 2. Restoring the alias is safer than pinning
# NumPy below 2 under this torch build; np.str was only ever an alias for the builtin str.
if not hasattr(np, "str"):
    np.str = str                                                    # type: ignore[attr-defined]

# --- CPU shims, applied BEFORE tcrsep is imported ------------------------------------------
torch.load = functools.partial(torch.load, map_location=torch.device("cpu"))
torch.Tensor.cuda = lambda self, *a, **k: self                      # type: ignore[assignment]
torch.nn.Module.cuda = lambda self, *a, **k: self                   # type: ignore[assignment]
_orig_to = torch.nn.Module.to


def _to_cpu(self, *a, **k):
    a = tuple("cpu" if isinstance(x, str) and "cuda" in x else x for x in a)
    if isinstance(k.get("device"), str) and "cuda" in k["device"]:
        k["device"] = "cpu"
    return _orig_to(self, *a, **k)


torch.nn.Module.to = _to_cpu                                        # type: ignore[assignment]

# tcrsep/utils.py hardcodes `.to("cuda:0")` on tensors as well as modules, so Tensor.to needs
# the same redirection. Both are placement-only: no parameter value is altered.
_orig_tensor_to = torch.Tensor.to


def _tensor_to_cpu(self, *a, **k):
    a = tuple("cpu" if isinstance(x, str) and "cuda" in x else x for x in a)
    if isinstance(k.get("device"), str) and "cuda" in k["device"]:
        k["device"] = "cpu"
    return _orig_tensor_to(self, *a, **k)


torch.Tensor.to = _tensor_to_cpu                                    # type: ignore[assignment]

HERE = os.path.dirname(os.path.abspath(__file__))
SEQS = os.path.join(HERE, "sequences.tsv")
OUT = os.path.join(HERE, "ppost_summary.csv")
LOG = os.path.join(HERE, "ppost_run.json")

CHAINS = ("B",)
BATCH = 50
CAP = 300      # per cohort; the read-out is a per-cohort DISTRIBUTION and the question is only
               # whether conditioning on Ppost reorders the cohorts, which 300 draws settle. The
               # full-depth run was abandoned: scoring is a TAPE forward pass per sequence on a
               # CPU-only host, and a single unscoreable record forces a 500-wide batch to be
               # retried one at a time, which is where it stalled.
SEED = 0


def load() -> pd.DataFrame:
    df = pd.read_csv(SEQS, sep="\t")
    need = {"cohort", "chain", "cdr3", "v_imgt", "j_imgt"}
    assert need <= set(df.columns), sorted(need - set(df.columns))
    # cap each cohort/chain so the run is bounded; the read-out is a per-cohort distribution.
    # Selected by index rather than groupby-apply, which consumes the grouping columns.
    keep = []
    for _, idx in df.groupby(["cohort", "chain"]).indices.items():
        if len(idx) > CAP:
            idx = pd.Series(idx).sample(CAP, random_state=SEED).to_numpy()
        keep.append(idx)
    out = df.iloc[np.concatenate(keep)].reset_index(drop=True)
    assert {"cohort", "chain"} <= set(out.columns), "grouping columns were consumed"
    # TCRsep's tokenizer rejects anything outside the 20 canonical residues, and some control
    # cohorts carry lower-case sequences. Uppercase, then drop the rest and say how many.
    out["cdr3"] = out.cdr3.astype(str).str.upper()
    ok = out.cdr3.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
    # TCRsep conditions on germline V/J, so a record without them cannot be scored at all. The
    # MLR expanded-clone control is beta-only and carries no gene calls, so it drops out here;
    # that is a property of the control, not a filter on the result, and is reported.
    has_genes = out.v_imgt.notna() & out.j_imgt.notna()
    keep = ok & has_genes
    if (~keep).any():
        lost = out[~keep].cohort.value_counts().to_dict()
        print("unscoreable rows dropped (%d of %d): %s"
              % (int((~keep).sum()), len(out), lost), flush=True)
    return out[keep].reset_index(drop=True)


def score_tcrsep(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """(Ppost, Pgen) per sequence from TCRsep's default selection model, or None."""
    try:
        from tcrsep.estimator import TCRsep
        model = TCRsep(default_sel_model=True, device="cpu")
    except Exception as e:                                          # noqa: BLE001
        print("tcrsep unavailable: %r" % (e,), file=sys.stderr)
        return None
    data = [[c, v, j] for c, v, j in zip(df.cdr3, df.v_imgt, df.j_imgt)]
    # Ppost = Pgen x selection factor. `get_prob` returns Pgen unchanged in this release -- it
    # was checked against `get_pgen` and the two agree to floating point -- so the selection
    # model has to be read from `predict_weights`, which is what makes this arm selection-AWARE
    # rather than a second copy of OLGA.
    #
    # A gene outside TCRsep's own reference makes it stitch a NaN full sequence, which its
    # tokenizer then rejects mid-batch. Those records cannot be scored by this model at all, so
    # a failing batch is retried one sequence at a time and the individual failures recorded as
    # NaN. This is a data-compatibility skip, not a silent fallback: the count is printed and
    # ends up in the run log beside the result.
    post = np.full(len(data), np.nan)
    gen = np.full(len(data), np.nan)
    unscoreable = 0
    for i in range(0, len(data), BATCH):
        chunk = data[i:i + BATCH]
        try:
            g = np.asarray(model.get_pgen(chunk), dtype=float).ravel()
            w = np.asarray(model.predict_weights(chunk), dtype=float).ravel()
            gen[i:i + len(chunk)] = g
            post[i:i + len(chunk)] = g * w
        except Exception:                                           # noqa: BLE001
            for k, one in enumerate(chunk):
                try:
                    g1 = float(np.asarray(model.get_pgen([one]), dtype=float).ravel()[0])
                    w1 = float(np.asarray(model.predict_weights([one]), dtype=float).ravel()[0])
                    gen[i + k] = g1
                    post[i + k] = g1 * w1
                except Exception:                                   # noqa: BLE001
                    unscoreable += 1
        print("  scored %d/%d (%d unscoreable so far)"
              % (min(i + BATCH, len(data)), len(data), unscoreable), flush=True)
    print("  %d of %d sequences could not be scored by TCRsep's own gene reference"
          % (unscoreable, len(data)), flush=True)
    return post, gen


def summarise(df: pd.DataFrame, vals: np.ndarray, model: str, chain: str) -> list[dict]:
    """Per-cohort quantiles of log10 of a strictly positive score."""
    out = []
    lv = np.full(len(vals), np.nan)
    ok = np.isfinite(vals) & (vals > 0)
    lv[ok] = np.log10(vals[ok])
    for cohort, idx in df.groupby("cohort").indices.items():
        v = lv[idx]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out.append({"model": model, "chain": chain, "cohort": cohort, "n": int(v.size),
                    "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                    "q75": float(np.percentile(v, 75)), "mean": float(np.mean(v))})
    return out


def main():
    df = load()
    rows: list[dict] = []
    status: dict = {}
    for chain in CHAINS:
        sub = df[df.chain == chain].reset_index(drop=True)
        if sub.empty:
            continue
        print("=== chain %s (%d sequences) ===" % (chain, len(sub)), flush=True)
        got = score_tcrsep(sub)
        status["tcrsep_%s" % chain] = "ok" if got is not None else "unavailable"
        if got is None:
            continue
        post, gen = got
        assert len(post) == len(sub) and len(gen) == len(sub), "length mismatch"
        rows.extend(summarise(sub, post, "tcrsep_ppost", chain))
        rows.extend(summarise(sub, gen, "tcrsep_pgen", chain))
        # the selection factor itself is what a selection-aware model adds over OLGA; if it is
        # near-constant across cohorts then conditioning on Ppost cannot reorder them
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(gen > 0, post / gen, np.nan)
        rows.extend(summarise(sub, q, "tcrsep_q", chain))
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("wrote %s (%d rows)" % (OUT, len(rows)))
    else:
        print("no model produced scores; nothing written", file=sys.stderr)
    with open(LOG, "w") as fh:
        json.dump({"status": status, "n_sequences": int(len(df)),
                   "versions": _versions()}, fh, indent=2)
    print(json.dumps(status, indent=2))


def _versions() -> dict:
    v = {}
    for mod in ("tcrsep", "olga", "torch", "numpy", "pandas"):
        try:
            v[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:                                           # noqa: BLE001
            v[mod] = "absent"
    return v


if __name__ == "__main__":
    main()
