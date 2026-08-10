"""Model-based signal probe: ESM-2 embedding separation of epitopes per cohort.

An orthogonal, learned-representation analogue of the (model-free) homology S/N.
For each cohort we embed CDR3s with ESM-2, then measure how much the epitope
labels structure the embedding space, one-vs-many per epitope:

  For each CDR3 we take its K_NN nearest neighbours (cosine) among the whole cohort
  and compute epitope purity = fraction of neighbours sharing its epitope. Per
  epitope, enrichment = mean purity / null purity (n_e/N under random neighbours);
  the cohort value is the geometric mean of enrichment over epitopes (>= MIN_N
  records), with a 95% CI (log SE) -- exactly the aggregation used for homology S/N.
  Signal => enrichment >> 1, noise => ~1.

Runs on Apple MPS or CPU (small ESM-2). Embeddings are cached per cohort/chain.
Usage: python src/esm_signal.py            # writes results/esm_signal.csv + .dat
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META   # noqa: E402

CACHE = os.path.join(REPO, "cache", "esm"); os.makedirs(CACHE, exist_ok=True)
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")

# Parameters shared by every ESM-based probe (esm_cluster, degenerate, mi_master import these,
# so there is one value per knob across the study). Tabulated in the supplement.
MODEL = "facebook/esm2_t12_35M_UR50D"   # 480-d; small + fast, and CDR3s are short
MIN_N = 30            # qualifying epitope size (matches the homology probe)
PER_EPI = 60          # cap sequences per epitope: equalises the epitope weights within a cohort
K_NN = 15             # neighbours, for BOTH the purity statistic and the Leiden graph
PCA_DIM = 12          # global PCA dimension for the kNN MI estimator (src/esm_pca.py)


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed(seqs: list[str], batch: int = 128) -> np.ndarray:
    """Mean-pooled ESM-2 residue embeddings, L2-normalized. One vector per CDR3."""
    import torch
    from transformers import AutoTokenizer, EsmModel
    dev = _device()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmModel.from_pretrained(MODEL).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
            rep = model(**enc).last_hidden_state          # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # mean over real residues, excluding the two special tokens (BOS/EOS)
            mask[:, 0, :] = 0
            for b, L in enumerate(enc["attention_mask"].sum(1)):
                mask[b, int(L) - 1, :] = 0
            vec = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(vec.float().cpu().numpy())
    X = np.concatenate(out, 0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def _subsample(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Up to PER_EPI receptors from EVERY epitope with >= MIN_N of them.

    All qualifying epitopes are kept. An earlier version also capped the epitope count, which cut
    the deeply characterised cohorts (VDJdb HQ TCRbeta: 83 qualifying epitopes) to the same 25 as
    the shallow ones while leaving the small cohorts whole -- an asymmetry with no justification.
    PER_EPI still equalises epitopes WITHIN a cohort, so no single public epitope dominates.
    """
    vc = df.epitope.value_counts()
    eps = vc[vc >= MIN_N].index.to_list()
    parts = [df[df.epitope == e].sample(min(PER_EPI, int(vc[e])), random_state=0) for e in eps]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def cohort_signal(name: str, sub: pd.DataFrame, X: np.ndarray) -> dict:
    """One-vs-many kNN epitope-purity enrichment, geomean over epitopes (+/-95% CI)."""
    from sklearn.neighbors import NearestNeighbors
    N = len(sub)
    epi = sub.epitope.to_numpy()
    nn = NearestNeighbors(n_neighbors=min(K_NN + 1, N), metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                                  # drop self
    same = (epi[idx] == epi[:, None]).mean(1)         # per-seq purity
    logs = []
    for e in pd.unique(epi):
        m = epi == e
        n_e = int(m.sum())
        null = (n_e - 1) / max(N - 1, 1)              # random-neighbour expectation
        enr = (same[m].mean() + 1e-3) / (null + 1e-3)
        logs.append(np.log(max(enr, 1e-6)))
    logs = np.asarray(logs)
    g = float(np.exp(logs.mean()))
    se = float(logs.std(ddof=1) / np.sqrt(len(logs))) if len(logs) > 1 else 0.0
    return {"sn": g, "lo": float(np.exp(logs.mean() - 1.96 * se)),
            "hi": float(np.exp(logs.mean() + 1.96 * se)), "n_ep": len(logs), "n": N}


def run():
    rng = np.random.default_rng(0)
    coh = build_cohorts(include_olga=True)
    order = [c for c in HIERARCHY if c in coh]
    rows = []
    for name in order:
        long = coh[name]["long"]
        for chain in ("A", "B"):
            lc = long[long.chain == chain]
            if name == "mlr_prolif" and chain == "A":
                continue
            sub = _subsample(lc[["epitope", "cdr3"]].dropna(), rng)
            if sub.epitope.nunique() < 2:
                continue
            cache = os.path.join(CACHE, f"{name}_{chain}.npy")
            seqs = sub.cdr3.to_list()
            if os.path.exists(cache) and np.load(cache).shape[0] == len(seqs):
                X = np.load(cache)
            else:
                X = embed(seqs); np.save(cache, X)
            s = cohort_signal(name, sub, X)
            rows.append({"cohort": name, "label": COHORT_META[name]["label"], "chain": chain, **s})
            print("%-16s TR%s  esmSN=%.2f [%.2f,%.2f]  n_ep=%d  n=%d" %
                  (name, chain, s["sn"], s["lo"], s["hi"], s["n_ep"], s["n"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "esm_signal.csv"), index=False)
    # .dat for the supplement bar figures, x = HIERARCHY index (both chains)
    idx = {n: i for i, n in enumerate(HIERARCHY)}
    for chain in ("B", "A"):
        d = df[df.chain == chain].copy(); d["x"] = [idx[c] for c in d.cohort]
        with open(os.path.join(ADAT, f"esm_signal_{chain}.dat"), "w") as fh:
            fh.write("# x cohort sn lo hi\n")
            d[["x", "cohort", "sn", "lo", "hi"]].to_csv(fh, sep=" ", index=False, header=False)
    print("\nwrote results/esm_signal.csv + appendix/analysis/esm_signal_{A,B}.dat")


if __name__ == "__main__":
    run()
