"""OLGA pgen-matched random control.

Build a synthetic cohort whose per-chain generation probability (pgen)
distribution matches the immrep25 positives, with alpha/beta paired at random
into synthetic epitope groups of the same shape as immrep (20 x 50). pgen
controls *random matchability*: if immrep positives match this floor, their
homology/pairing is explained by generation statistics, not epitope selection.

Uses the OLGA Python API (human_T_alpha = VJ recombination, human_T_beta = VDJ).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import olga.load_model as load_model
import olga.generation_probability as gen_prob
import olga.sequence_generation as seq_gen

_OLGA_DIR = os.path.dirname(load_model.__file__)


def _model_paths(chain: str):
    folder = "human_T_alpha" if chain == "A" else "human_T_beta"
    base = os.path.join(_OLGA_DIR, "default_models", folder)
    return (os.path.join(base, "model_params.txt"),
            os.path.join(base, "model_marginals.txt"),
            os.path.join(base, "V_gene_CDR3_anchors.csv"),
            os.path.join(base, "J_gene_CDR3_anchors.csv"))


def load_models(chain: str):
    """Return (pgen_model, seqgen_model, genomic_data) for chain 'A' (VJ) or 'B' (VDJ)."""
    params, marginals, vanc, janc = _model_paths(chain)
    if chain == "A":
        gd = load_model.GenomicDataVJ(); gm = load_model.GenerativeModelVJ()
        gd.load_igor_genomic_data(params, vanc, janc)
        gm.load_and_process_igor_model(marginals)
        return (gen_prob.GenerationProbabilityVJ(gm, gd),
                seq_gen.SequenceGenerationVJ(gm, gd), gd)
    gd = load_model.GenomicDataVDJ(); gm = load_model.GenerativeModelVDJ()
    gd.load_igor_genomic_data(params, vanc, janc)
    gm.load_and_process_igor_model(marginals)
    return (gen_prob.GenerationProbabilityVDJ(gm, gd),
            seq_gen.SequenceGenerationVDJ(gm, gd), gd)


def compute_pgen(seqs, chain: str) -> np.ndarray:
    pm, _, _ = load_models(chain)
    out = np.empty(len(seqs))
    for i, s in enumerate(seqs):
        try:
            out[i] = pm.compute_aa_CDR3_pgen(s)
        except Exception:
            out[i] = 0.0
    return out


def generate_pool(chain: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Generate n productive CDR3s with V/J and their pgen."""
    pm, sg, gd = load_models(chain)
    np.random.seed(seed)
    Vnames = [g[0] for g in gd.genV]
    Jnames = [g[0] for g in gd.genJ]
    cdr3, v, j, pg = [], [], [], []
    for _ in range(n):
        nt, aa, vi, ji = sg.gen_rnd_prod_CDR3()
        cdr3.append(aa); v.append(Vnames[vi].split("*")[0]); j.append(Jnames[ji].split("*")[0])
        try:
            pg.append(pm.compute_aa_CDR3_pgen(aa))
        except Exception:
            pg.append(0.0)
    return pd.DataFrame({"cdr3": cdr3, "v": v, "j": j, "pgen": pg})


def pgen_match(target_pgen: np.ndarray, pool: pd.DataFrame, n_target: int,
               rng, n_bins: int = 25) -> pd.DataFrame:
    """Sample n_target pool rows so their pgen histogram matches target_pgen."""
    tp = target_pgen[target_pgen > 0]
    pool = pool[pool.pgen > 0].copy()
    lt = np.log10(tp)
    lp = np.log10(pool.pgen.values)
    edges = np.quantile(lt, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tgt_hist, _ = np.histogram(lt, bins=edges)
    frac = tgt_hist / tgt_hist.sum()
    pool_bin = np.digitize(lp, edges) - 1
    picks = []
    for b in range(n_bins):
        want = int(round(frac[b] * n_target))
        cand = np.where(pool_bin == b)[0]
        if want == 0 or cand.size == 0:
            continue
        picks.append(rng.choice(cand, want, replace=cand.size < want))
    idx = np.concatenate(picks) if picks else np.array([], int)
    if len(idx) < n_target:                       # top up rounding shortfall -> exactly n_target
        idx = np.concatenate([idx, rng.choice(len(pool), n_target - len(idx), replace=True)])
    rng.shuffle(idx)
    return pool.iloc[idx[:n_target]].reset_index(drop=True)


def build_olga_control(immrep_paired: pd.DataFrame, n_pool: int = 40000,
                       seed: int = 0) -> pd.DataFrame:
    """Return a paired OLGA cohort (cdr3a,va,ja,cdr3b,vb,jb,epitope) matched to immrep."""
    rng = np.random.default_rng(seed)
    pa = compute_pgen(immrep_paired.cdr3a.tolist(), "A")
    pb = compute_pgen(immrep_paired.cdr3b.tolist(), "B")
    pool_a = generate_pool("A", n_pool, seed=seed)
    pool_b = generate_pool("B", n_pool, seed=seed + 1)
    n = len(immrep_paired)
    ma = pgen_match(pa, pool_a, n, rng)
    mb = pgen_match(pb, pool_b, n, rng)
    n = min(len(ma), len(mb))
    ma, mb = ma.iloc[:n], mb.iloc[:n]
    # random alpha<->beta pairing; synthetic epitope groups matching immrep shape
    order = rng.permutation(n)
    n_ep = immrep_paired.epitope.nunique()
    grp = np.arange(n) % n_ep
    out = pd.DataFrame({
        "cohort": "olga_random",
        "tcr_id": ["olga_%d" % i for i in range(n)],
        "epitope": ["olga_ep%02d" % g for g in grp],
        "cdr3a": ma.cdr3.values, "va": ma.v.values, "ja": ma.j.values,
        "cdr3b": mb.cdr3.values[order], "vb": mb.v.values[order], "jb": mb.j.values[order],
    })
    return out, {"pa": pa, "pb": pb, "pool_a": pool_a, "pool_b": pool_b}


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "src")
    from cohorts import build_cohorts
    imm = build_cohorts()["immrep25_pos"]["paired"]
    t = time.time()
    pa = compute_pgen(imm.cdr3a.tolist()[:50], "A")
    pb = compute_pgen(imm.cdr3b.tolist()[:50], "B")
    pool_b = generate_pool("B", 2000, seed=0)
    print("timing(50a+50b pgen + 2000b gen): %.1fs" % (time.time() - t))
    print("immrep alpha log10 pgen: med=%.2f q10=%.2f q90=%.2f" %
          (np.median(np.log10(pa[pa > 0])), np.quantile(np.log10(pa[pa > 0]), .1), np.quantile(np.log10(pa[pa > 0]), .9)))
    print("immrep beta  log10 pgen: med=%.2f q10=%.2f q90=%.2f" %
          (np.median(np.log10(pb[pb > 0])), np.quantile(np.log10(pb[pb > 0]), .1), np.quantile(np.log10(pb[pb > 0]), .9)))
    gp = pool_b[pool_b.pgen > 0]
    print("OLGA gen beta log10 pgen: med=%.2f q10=%.2f q90=%.2f" %
          (np.median(np.log10(gp.pgen)), np.quantile(np.log10(gp.pgen), .1), np.quantile(np.log10(gp.pgen), .9)))
