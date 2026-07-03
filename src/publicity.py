"""Publicity control — is immrep25's (already weak) homology signal just public TCRs?

Flags each immrep positive whose CDR3 exactly- or Hamming<=1-matches a known TCR
in VDJdb/TCRvdb, then re-runs the homology S/N with public TCRs removed. If S/N is
unchanged and stays at the floor, residual signal (if any) is publicity, not
epitope-specific biology.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


def _wildcards(seq: str):
    return (seq, *[(len(seq), p, seq[:p] + "*" + seq[p + 1:]) for p in range(len(seq))])


def build_reference_index(ref_seqs) -> tuple[set, set]:
    """Return (exact set, wildcard-pattern set) for near-neighbour membership tests."""
    exact, patt = set(), set()
    for s in ref_seqs:
        if not isinstance(s, str) or len(s) < 5:
            continue
        exact.add(s)
        for p in range(len(s)):
            patt.add((len(s), p, s[:p] + "*" + s[p + 1:]))
    return exact, patt


def is_public(seq: str, exact: set, patt: set) -> bool:
    if not isinstance(seq, str):
        return False
    if seq in exact:
        return True
    for p in range(len(seq)):
        if (len(seq), p, seq[:p] + "*" + seq[p + 1:]) in patt:
            return True
    return False


def annotate_publicity(immrep_pos: pd.DataFrame, vdjdb_long: pd.DataFrame,
                       tcrvdb_paired: list[pd.DataFrame]) -> pd.DataFrame:
    """Add public_a / public_b / public_any flags (exact or Hamming<=1 vs references)."""
    ref_a = list(vdjdb_long[vdjdb_long.chain == "A"].cdr3)
    ref_b = list(vdjdb_long[vdjdb_long.chain == "B"].cdr3)
    for tp in tcrvdb_paired:
        ref_a += list(tp.cdr3a); ref_b += list(tp.cdr3b)
    ea, pa = build_reference_index(ref_a)
    eb, pb = build_reference_index(ref_b)
    out = immrep_pos.copy()
    out["public_a"] = [is_public(s, ea, pa) for s in out.cdr3a]
    out["public_b"] = [is_public(s, eb, pb) for s in out.cdr3b]
    out["public_any"] = out.public_a | out.public_b
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from cohorts import build_cohorts
    from load_data import load_vdjdb
    from homology import homology_sn_bootstrap, summarize
    coh = build_cohorts()
    imm = coh["immrep25_pos"]["paired"]
    vlong, _ = load_vdjdb()
    ann = annotate_publicity(imm, vlong, [coh["tcrvdb_true"]["paired"], coh["tcrvdb_false"]["paired"]])
    print("immrep positives flagged public (exact or Ham<=1 vs VDJdb+TCRvdb):")
    print("  alpha:%.1f%%  beta:%.1f%%  any:%.1f%%  both:%.1f%%" % (
        100 * ann.public_a.mean(), 100 * ann.public_b.mean(),
        100 * ann.public_any.mean(), 100 * (ann.public_a & ann.public_b).mean()))

    from load_data import _pair_to_long
    for label, sub in [("all", ann), ("non-public only", ann[~ann.public_any])]:
        lg = _pair_to_long(sub)
        for chain in ("B", "A"):
            lc = lg[lg.chain == chain]
            boot = homology_sn_bootstrap(lc, K=30, E_cap=2, n_boot=60, max_d=3, seed=1)
            r = summarize(boot); r = r[r.d == 1].iloc[0]
            print(f"  homology d1 S/N [{label:<16} TR{chain}] = {r.sn_median:5.2f} [{r.sn_lo:.2f},{r.sn_hi:.2f}]  (n={len(sub)})")
