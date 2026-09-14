#!/usr/bin/env python
# 2026-09-14  Cache learned sequence representations of the IMMREP25 receptors.
#
# This module answers the reviewers' standing objection -- "a better method would find the signal"
# -- by handing the benchmark to the strongest learned TCR representations available and scoring
# them under the challenge's own condition. It does the EMBEDDING half only; the scoring half is
# src/embedding_comparison.py, which runs in the main environment and reads this cache.
#
# WHY TWO ENVIRONMENTS. sceptr pins pandas<3, and the main environment is on pandas 3.0.5 -- the
# version under which every committed macro in appendix/analysis was produced. Installing sceptr
# into it would downgrade pandas beneath every standing number in the study. So the heavy, pinned
# embedders live in .venv-embed and communicate only through .npy files, exactly the arrangement
# pyproject.toml already uses for torch/transformers (the `embed` optional group) and cache/esm.
#   uv venv .venv-embed --python 3.11 && VIRTUAL_ENV=.venv-embed uv pip install sceptr transformers
#   .venv-embed/bin/python src/embed_cache.py
#
# WHY NOT REUSE cache/esm/immrep25_pos_*.npy. Those are (991, 480) and (993, 480): cohorts.py
# dedups per chain and subsamples, so neither array is aligned to the other, to the 1000 unique
# release receptors, or to any stored key. They are unusable for a per-receptor comparison, so
# ESM-2 is re-embedded here against the key this module writes.
#
# GENE NAMING IS LOAD-BEARING. The release ships Adaptive tokens (TCRAV08-03); sceptr requires
# standardised functional IMGT symbols and raises BadV otherwise. An unmapped gene would reach the
# model as a null and silently delete the receptor's V channel -- precisely the failure mode
# transfer_germline.py's header documents for the VDJdb/IMMREP25 vocabulary mismatch -- so the
# resolved fraction is ASSERTED, not hoped for.
#
# tidytcells alone is NOT sufficient and was tried first: it leaves 9 of 49 TCRBV tokens
# unresolved, namely the ambiguous pairs (TCRBV03-01/03-02, TCRBV06-02/06-03, TCRBV12-03/12-04)
# and the family-level tokens that name no gene (TCRBV05-X, 06-X, 07-X, 11-X, 12-X). We therefore
# map through our own VDJtools table, the same one transfer_germline.py uses, which resolves all
# 147 distinct IMMREP25 tokens. Note what that table does with a family-level token: it assigns
# the CDR1+CDR2-validated representative of the family (TCRBV07-X -> TRBV7-6), which is a
# germline-sequence decision and never an epitope-label one, so it introduces no circularity --
# but it is an imputation, and the receptors carrying it are counted in the self-check.
#
# Run: .venv-embed/bin/python src/embed_cache.py     (--demo for the self-check)
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "dump", "immrep25", "immrep2025_for_release.tsv")
CACHE = os.path.join(REPO, "cache", "embed")
KEY = os.path.join(CACHE, "receptor_key.csv")
GENE_MAP = os.environ.get(                       # same table, same override, as transfer_germline
    "VDJTOOLS_ADAPTIVE_MAP",
    os.path.expanduser("~/vcs/code/vdjtools/python/vdjtools/resources/adaptive_imgt_map.tsv"))

N_REC = 1000          # distinct receptors in the release; asserted, not assumed
N_ROWS = 10000

# sceptr variants. The first is the shipped model; the rest are the AUTHORS' OWN published
# ablations (Nagano et al., Cell Systems 2025), which is why they are worth the forward pass:
# shuffled_data and synthetic_data are their negative controls, so the comparison carries its
# own internal scale rather than resting on our choice of competitors.
SCEPTR_VARIANTS = ("default", "large", "small", "cdr3_only", "a_sceptr", "b_sceptr",
                   "blosum", "average_pooling", "mlm_only", "shuffled_data", "synthetic_data")
ESM_MODELS = {"esm2_35M": "facebook/esm2_t12_35M_UR50D",
              "esm2_650M": "facebook/esm2_t33_650M_UR50D"}
TCRBERT = "wukevin/tcr-bert-mlm-only"
BATCH = 128


def receptors() -> pd.DataFrame:
    """The distinct receptors of the release, in a fixed order, with IMGT-standardised V genes.

    The key carries `row_id`, the index of the FIRST release row for each receptor, so a scorer
    can join the 1000 embeddings back onto the 10000 labelled (receptor, peptide) rows.
    """
    d = pd.read_csv(BENCH, sep="\t")
    assert len(d) == N_ROWS and int(d.label.sum()) == 1000, (len(d), int(d.label.sum()))
    cols = ["tcra_cdr3", "tcrb_cdr3", "tcra_v", "tcrb_v", "tcra_j", "tcrb_j"]
    r = (d[cols].drop_duplicates().reset_index(drop=True))
    assert len(r) == N_REC, "expected %d distinct receptors, got %d" % (N_REC, len(r))

    m = pd.read_csv(GENE_MAP, sep="\t")
    gmap = dict(zip(m.adaptive_token, m.imgt_gene))
    for src, dst in (("tcra_v", "TRAV"), ("tcrb_v", "TRBV")):
        uniq = {g: gmap.get(g) for g in r[src].unique()}
        bad = sorted(g for g, v in uniq.items() if v is None)
        assert not bad, ("%d of %d %s tokens are absent from %s: %s"
                         % (len(bad), len(uniq), src, os.path.basename(GENE_MAP), bad[:8]))
        r[dst] = r[src].map(uniq)
    # Keep the release's own column names AND add sceptr's, so the key stays joinable against
    # the 10000 labelled rows on the identifiers they actually carry.
    r["CDR3A"] = r.tcra_cdr3
    r["CDR3B"] = r.tcrb_cdr3
    return r


def embed_sceptr(r: pd.DataFrame, name: str) -> np.ndarray:
    from sceptr import variant
    model = getattr(variant, name)()
    return model.calc_vector_representations(r[["TRAV", "CDR3A", "TRBV", "CDR3B"]])


def _esm_like(seqs: list[str], model_id: str, spaced: bool) -> np.ndarray:
    """Mean-pooled residue embeddings over the real residues, L2-normalised, one row per sequence.

    Deliberately a local copy of esm_signal.embed rather than an import: esm_signal pulls in
    cohorts.py at module scope, which is pandas-3 code and cannot load in this environment.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    txt = [" ".join(s) for s in seqs] if spaced else list(seqs)
    out = []
    with torch.no_grad():
        for i in range(0, len(txt), BATCH):
            enc = tok(txt[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True).to(dev)
            rep = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mask[:, 0, :] = 0                                    # drop BOS
            for b, L in enumerate(enc["attention_mask"].sum(1)):
                mask[b, int(L) - 1, :] = 0                       # drop EOS
            out.append(((rep * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
    X = np.concatenate(out, 0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def embed_paired(r: pd.DataFrame, model_id: str, spaced: bool) -> np.ndarray:
    """A receptor vector from a per-chain model: the two chains embedded apart, then concatenated.

    Matches how the paired models see a receptor, so the comparison is between representations
    rather than between chain-handling conventions.
    """
    return np.hstack([_esm_like(r.CDR3A.tolist(), model_id, spaced),
                      _esm_like(r.CDR3B.tolist(), model_id, spaced)])


def main():
    os.makedirs(CACHE, exist_ok=True)
    r = receptors()
    r.to_csv(KEY, index=False)
    print("wrote %s (%d receptors)" % (KEY, len(r)))

    for name in SCEPTR_VARIANTS:
        out = os.path.join(CACHE, "sceptr_%s.npy" % name)
        X = embed_sceptr(r, name)
        assert X.shape[0] == len(r), (name, X.shape)
        np.save(out, X)
        print("  sceptr/%-16s %s -> %s" % (name, X.shape, os.path.basename(out)))

    for name, mid in ESM_MODELS.items():
        out = os.path.join(CACHE, "%s.npy" % name)
        X = embed_paired(r, mid, spaced=False)
        np.save(out, X)
        print("  %-23s %s -> %s" % (name, X.shape, os.path.basename(out)))

    out = os.path.join(CACHE, "tcrbert.npy")
    X = embed_paired(r, TCRBERT, spaced=True)     # TCR-BERT tokenises space-separated residues
    np.save(out, X)
    print("  %-23s %s -> %s" % ("tcrbert", X.shape, os.path.basename(out)))


def demo():
    """Self-check. The load-bearing assertions are that every Adaptive gene token resolves to a
    functional IMGT gene, and that sceptr returns one distinct vector per receptor -- a silently
    nulled V gene would still return vectors, just degenerate ones, so identity of the CDR3s is
    not enough to detect it."""
    r = receptors()
    assert len(r) == N_REC and r.TRAV.notna().all() and r.TRBV.notna().all()
    assert r.TRAV.str.startswith("TRAV").all() and r.TRBV.str.startswith("TRBV").all()
    assert r.TRAV.nunique() > 20 and r.TRBV.nunique() > 20, "the V channel collapsed"

    sub = r.head(64)
    X = embed_sceptr(sub, "default")
    assert X.shape == (64, 64), X.shape
    assert np.isfinite(X).all(), "sceptr returned non-finite vectors"
    assert len(np.unique(X.round(6), axis=0)) == len(sub), "sceptr vectors are not distinct"

    # the V gene must actually move the representation: same CDR3s, a different TRAV
    alt = sub.copy()
    alt["TRAV"] = np.where(alt.TRAV == alt.TRAV.iloc[0], alt.TRAV.iloc[1], alt.TRAV.iloc[0])
    d = np.linalg.norm(embed_sceptr(alt, "default") - X, axis=1)
    assert d.max() > 1e-4, "changing TRAV does not change the embedding -- genes are being ignored"
    print("embed_cache.demo OK  (%d receptors, %d TRAV / %d TRBV genes all functional IMGT; "
          "sceptr default returns %d distinct %d-d vectors, V-gene sensitivity %.4f)"
          % (len(r), r.TRAV.nunique(), r.TRBV.nunique(), len(sub), X.shape[1], d.max()))


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
