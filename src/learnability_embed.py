#!/usr/bin/env python
# 2026-09-15  The embedding half of src/learnability.py. Runs in the PINNED environment.
#
# WHY A SEPARATE FILE AND A SEPARATE ENVIRONMENT. Measured, not assumed: .venv-embed carries
# sceptr 1.2.0 + torch + transformers but NO scikit-learn and NO polars, while .venv carries
# sklearn + polars + pandas 3.0.5 but NO sceptr (which pins pandas~=2.0, beneath every committed
# macro in this study). So neither environment can run the whole probe. This file therefore does
# ONLY forward passes and communicates through .npy, exactly the arrangement embed_cache.py
# already uses -- and it reads plain CSVs written by --prep rather than importing load_data, so
# no pandas-3 code has to execute under pandas 2.3.3.
#
# ROW ORDER IS THE CONTRACT. Each .npy is positionally aligned to the CSV it was built from, and
# the alignment is asserted here rather than trusted downstream, as embed_cache.py does with its
# receptor_key.csv.
#
# TWO THINGS NOT TO "SIMPLIFY":
#
# 1. PER-CHAIN L2 NORMALISATION, THEN CONCATENATION. embed_cache.embed_paired L2-normalises each
#    chain's ESM vector separately and then hstacks them, so the cached IMMREP25 matrices
#    (cache/embed/esm2_35M.npy, 1000 x 960) are half-wise unit-norm. The OLGA background basis is
#    APPLIED to those cached matrices, so it must be fitted on vectors built the same way. Pooling
#    the two chains before normalising would put the basis on a different scale and silently
#    mis-project the benchmark.
# 2. UNRESOLVABLE V GENES FAIL LOUDLY. sceptr raises BadV on a non-standard symbol, and an
#    unmapped gene reaching the model as a null would silently delete that receptor's V channel --
#    the failure embed_cache.py's header documents. learnability.py canonicalises the three TRAV
#    respellings and drops the one non-functional gene before writing rec_vdjdb.csv, so this
#    should pass; if it does not, the offending tokens are printed and the run aborts rather than
#    embedding a degraded receptor.
#
# Run: .venv-embed/bin/python src/learnability_embed.py
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Must match src/learnability.py's WORK. Kept as a bare env lookup rather than an import because
# learnability.py imports sklearn, which this environment does not have.
WORK = os.environ.get("LEARNABILITY_WORK", os.path.expanduser("~/tmp/learnability"))
ESM = "facebook/esm2_t12_35M_UR50D"      # esm_signal.MODEL; 480-d per chain, weights already local
BATCH = 128


def _esm_like(seqs: list[str]) -> np.ndarray:
    """Mean-pooled residue embeddings over the real residues, L2-normalised, one row per sequence.

    A local copy of embed_cache._esm_like (itself a deliberate copy of esm_signal.embed) so this
    file imports nothing from src/ and stays loadable under pandas 2.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ESM)
    model = AutoModel.from_pretrained(ESM).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            enc = tok(list(seqs[i:i + BATCH]), return_tensors="pt", padding=True,
                      truncation=True).to(dev)
            rep = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mask[:, 0, :] = 0                                    # drop BOS
            for b, L in enumerate(enc["attention_mask"].sum(1)):
                mask[b, int(L) - 1, :] = 0                       # drop EOS
            out.append(((rep * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
    X = np.concatenate(out, 0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def embed_paired_esm(r: pd.DataFrame) -> np.ndarray:
    """Two chains embedded apart, each L2-normalised, then concatenated -- see note 1."""
    return np.hstack([_esm_like(r.CDR3A.tolist()), _esm_like(r.CDR3B.tolist())])


def embed_sceptr(r: pd.DataFrame) -> np.ndarray:
    """SCEPTR's own paired input: TRAV + CDR3A + TRBV + CDR3B (no J reaches the model)."""
    bad = sorted({t for c in ("TRAV", "TRBV") for t in r[c].astype(str).unique()
                  if not t.startswith(("TRAV", "TRBV"))})
    assert not bad, "non-IMGT V tokens, sceptr would raise BadV: %s" % bad[:12]
    from sceptr import variant
    model = variant.default()
    try:
        return model.calc_vector_representations(r[["TRAV", "CDR3A", "TRBV", "CDR3B"]])
    except Exception as exc:
        print("sceptr failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        print("distinct TRAV: %s" % sorted(r.TRAV.astype(str).unique())[:20], file=sys.stderr)
        print("distinct TRBV: %s" % sorted(r.TRBV.astype(str).unique())[:20], file=sys.stderr)
        raise


def save(X: np.ndarray, n: int, tag: str):
    assert X.shape[0] == n, (tag, X.shape, n)
    assert np.isfinite(X).all(), "%s: non-finite embedding" % tag
    out = os.path.join(WORK, "emb_%s.npy" % tag)
    np.save(out, X)
    print("  %-22s %s -> %s" % (tag, X.shape, os.path.basename(out)))


def main():
    assert os.path.isdir(WORK), \
        "run .venv/bin/python src/learnability.py --prep first (LEARNABILITY_WORK=%s)" % WORK

    for name in ("vdjdb", "olgabg"):
        r = pd.read_csv(os.path.join(WORK, "rec_%s.csv" % name))
        print("%s: %d paired receptors" % (name, len(r)))
        save(embed_sceptr(r), len(r), "rec_%s_sceptr" % name)
        save(embed_paired_esm(r), len(r), "rec_%s_esm" % name)

    for name in ("panel", "bg"):
        p = pd.read_csv(os.path.join(WORK, "pep_%s.csv" % name))
        print("peptides/%s: %d sequences" % (name, len(p)))
        save(_esm_like(p.peptide.tolist()), len(p), "pep_%s_esm" % name)

    print("\nnow: .venv/bin/python src/learnability.py --score")


if __name__ == "__main__":
    main()
