"""Structural-convergence probe: does epitope structure survive in the folded interface?
# 2026-08-06

A structural analogue of the homology S/N and ESM clustering probes, using tcren
(recognition_features, Karnaukhov 2024) purely as a feature EXTRACTOR -- not its predictor.
For each cohort we take the per-complex TCR-interface descriptor vector (docking pose,
CDR3-loop energies, CDR3<->peptide contact chemistry incl. the hydrophobic channel, and the
CDR3 groove-frame strain terms), z-score within cohort, and measure whether TCRs assigned the
same epitope have more similar interfaces than TCRs of other epitopes -- one-vs-many, exactly
as for sequence homology.

Scaffold confound: within one epitope group the peptide+MHC are fixed, so ANY descriptor that
is a function of the pMHC scaffold separates epitopes trivially (in every cohort, including the
mismatched-TCR negatives). We therefore (a) keep only TCR/CDR3-interface descriptors, dropping
pMHC-scaffold and QC columns, and (b) VERIFY the removal with the negative floor: mismatched
TCRs must give S/N ~ 1. If the floor is not ~1, a descriptor is leaking scaffold.

Cohorts: signal calibrators = vdjdb_binder (real binders, template-covered subset) and
immrep22_rest (same AF pipeline as IMMREP25); system under test = IMMREP25 positives;
floor + leak-test = immrep_2022 mismatched negatives (label=0). All structures are predicted,
which is the point: predicted structure is a function of sequence, so this cannot exceed the
sequence-level epitope information (Result 3 / data-processing inequality in the manuscript).

Usage: python src/structure_signal.py   # writes results/struct_convergence.csv + .dat + macros
Reads results/struct_desc_<cohort>.tsv (produced by `tcren recognize --features-only --full`).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import igraph as ig
import leidenalg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi import mi_report                                                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
ADAT = os.path.join(REPO, "appendix", "analysis")
CACHE = os.path.join(REPO, "cache", "struct")
HF = os.path.expanduser("~/hf/tcren_structures")

MIN_N = 15          # qualifying epitope size (structures are scarcer than sequences)
IPTM_MIN = 0.5      # AlphaFold interface-confidence filter (tcr-pmhc_iptm)
K = 10              # neighbours for the S/N purity probe
KNN = 15            # neighbours for the Leiden graph
RES = 1.0
N_BOOT = 1000       # bootstrap resamples (n structures with replacement) for the metric CIs
N_PERM = 1000       # permutation null for MI-in-bits (mi.mi_report)

# ---- Descriptor sets --------------------------------------------------------------------------
# FULL: all TCR/CDR3-interface descriptors (pMHC-scaffold + QC columns already excluded).
#   `crossing` dropped -- it duplicates `crossing_signed` under (cos,sin) (M4).
# CDR3ONLY: strictly TCR-conformation terms (CDR3 groove-frame strain + CDR3-loop energies +
#   chain balance) -- NO docking pose and NO peptide-dependent contact chemistry. This is the
#   genuinely scaffold-free set used to VERIFY the mismatched-negative floor -> S/N ~ 1 (M3).
ANGLES = ["pitch", "crossing_signed", "dock_torsion"]                       # -> cos/sin
GEOM = ["dock_d", "dock_tcr_uy", "dock_tcr_uz"]
ENER = ["F_cdr12", "F_cdr3a", "F_cdr3b", "F_tcr_pep", "dF_tcr_pep"]         # TCR-peptide energies
CHEM = ["ct_tp_salt_bridge", "ct_tp_aromatic", "ct_tp_hydrophobic", "ct_tp_other",
        "n_contacts_tp", "n_pep_contacted", "chain_balance"]               # CDR3<->peptide chemistry
STRAIN = [f"cdr3{c}_{s}" for c in "ab"
          for s in ["reach", "ou", "ow", "on", "au", "aw", "an", "topep", "ext"]]
LINEAR = GEOM + ENER + CHEM + STRAIN
CDR3ONLY_LINEAR = STRAIN + ["F_cdr3a", "F_cdr3b", "chain_balance"]
CDR3ONLY_ANGLES: list[str] = []
HYD = "ct_tp_hydrophobic"          # the literal hydrophobic-contact channel (reported explicitly)


def _feature_matrix(df: pd.DataFrame, cdr3_only: bool = False) -> np.ndarray:
    """z-scored TCR-interface vector; circular angles entered as (cos,sin) in radians.

    Per-column NaN imputation (m6): fill each column with its own nanmean BEFORE z-scoring,
    so a missing value becomes the column mean (a zero after standardization), not a global
    cross-feature outlier.
    """
    lin = CDR3ONLY_LINEAR if cdr3_only else LINEAR
    ang = CDR3ONLY_ANGLES if cdr3_only else ANGLES
    cols = []
    for c in lin:
        if c in df:
            v = df[c].to_numpy(dtype=float)
            mu = np.nanmean(v) if np.isfinite(np.nanmean(v)) else 0.0
            cols.append(np.where(np.isnan(v), mu, v))
    for c in ang:
        if c in df:
            a = np.deg2rad(df[c].to_numpy(dtype=float))
            for ch in (np.cos(a), np.sin(a)):
                mu = np.nanmean(ch) if np.isfinite(np.nanmean(ch)) else 0.0
                cols.append(np.where(np.isnan(ch), mu, ch))
    X = np.column_stack(cols)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd


# ---- epitope + ipTM attachment per cohort ----------------------------------------------------
def _stats_iptm(cohort_dir: str, complex_id: str) -> float:
    f = os.path.join(cohort_dir, complex_id + ".stats.json")
    if not os.path.exists(f):
        return np.nan
    try:
        return float(json.load(open(f))["ranked_0"]["tcr-pmhc_iptm"])
    except Exception:
        return np.nan


def load_cohort(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS, f"struct_desc_{name}.tsv"), sep="\t")
    df = df.rename(columns={"complex.id": "cid"})
    if name == "immrep25":
        emap = {}
        for line in open(os.path.join(CACHE, "immrep25_epitopes.tsv")):
            parts = line.rstrip("\n").replace("\\t", "\t").split("\t")
            if len(parts) >= 2:
                emap[parts[0]] = parts[1]
        df["epitope"] = df.cid.map(emap)
        df["iptm"] = [_stats_iptm(os.path.join(CACHE, "immrep25"), c) for c in df.cid]
        df["label"] = 1
    elif name == "immrep22rest":
        df["epitope"] = df.cid.str.split("_").str[2]
        df["iptm"] = [_stats_iptm(os.path.join(CACHE, "immrep22rest"), c) for c in df.cid]
        df["label"] = 1
    elif name == "negatives":
        df["epitope"] = df.cid.str.split("_").str[2]
        meta = pd.read_csv(os.path.join(HF, "immrep23_negatives", "immrep2022_negatives.tsv"),
                           sep="\t")
        key = (meta.cdr3a.astype(str) + "_" + meta.cdr3b.astype(str))
        im = dict(zip(key, meta["tcr-pmhc_iptm"]))
        df["iptm"] = [im.get("_".join(c.split("_")[:2]), np.nan) for c in df.cid]
        df["label"] = 0
    elif name == "vdjdb_binder":
        meta = pd.read_csv(os.path.join(HF, "vdjdb_binder_benchmark", "metadata.tsv"), sep="\t")
        m = meta.drop_duplicates("id").set_index("id")
        df["epitope"] = df.cid.map(m["epitope"])
        df["iptm"] = df.cid.map(m["iptm"])
        df["label"] = df.cid.map(m["y"])
        df["covered"] = df.cid.map(m["native_structure_exists"]).astype(str).str.lower() == "true"
    return df


def _qualify(df: pd.DataFrame) -> pd.DataFrame:
    """ipTM filter (where known) + keep epitopes with >= MIN_N structures."""
    n0 = len(df)
    df = df[df.iptm.isna() | (df.iptm >= IPTM_MIN)].copy()
    vc = df.epitope.value_counts()
    keep = vc[vc >= MIN_N].index
    df = df[df.epitope.isin(keep)].copy()
    df.attrs["dropped_iptm"] = n0 - len(df)
    return df


# ---- metrics (mirror esm_signal.cohort_signal + esm_cluster) ---------------------------------
def sn(df: pd.DataFrame, X: np.ndarray) -> dict:
    N = len(df); epi = df.epitope.to_numpy()
    nn = NearestNeighbors(n_neighbors=min(K + 1, N), metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X); idx = idx[:, 1:]
    same = (epi[idx] == epi[:, None]).mean(1)
    logs = []
    for e in pd.unique(epi):
        m = epi == e; n_e = int(m.sum())
        null = (n_e - 1) / max(N - 1, 1)
        logs.append(np.log(max((same[m].mean() + 1e-3) / (null + 1e-3), 1e-6)))
    logs = np.asarray(logs)
    se = float(logs.std(ddof=1) / np.sqrt(len(logs))) if len(logs) > 1 else 0.0
    return {"sn": float(np.exp(logs.mean())),
            "sn_lo": float(np.exp(logs.mean() - 1.96 * se)),
            "sn_hi": float(np.exp(logs.mean() + 1.96 * se)),
            "n_ep": len(logs), "n": N}


def _partition(X, seed=0):
    n = len(X)
    nn = NearestNeighbors(n_neighbors=min(KNN + 1, n), metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    edges = [(i, int(j)) for i, row in enumerate(idx[:, 1:]) for j in row]
    g = ig.Graph(n=n, edges=edges, directed=False); g.simplify()
    return np.asarray(leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=RES,
        seed=seed).membership)


def ari_ami(X, yt, seed=0):
    p = _partition(X, seed)
    return float(adjusted_rand_score(yt, p)), float(adjusted_mutual_info_score(yt, p))


def _boot_arrays(X, yt, rng):
    """Bootstrap ARI and AMI arrays: B resamples of n structures drawn with replacement."""
    aris, amis = [], []
    for b in range(N_BOOT):
        s = rng.choice(len(X), len(X), replace=True)
        if len(set(yt[s])) < 2:
            continue
        a, m = ari_ami(X[s], yt[s], seed=b)
        aris.append(a); amis.append(m)
    return np.asarray(aris), np.asarray(amis)


def _ci(point, boot):
    """Bootstrap 95% CI: recentre the bootstrap deviations around the full-sample point."""
    if not len(boot):
        return (np.nan, np.nan)
    dev = boot - boot.mean()
    return float(point + np.percentile(dev, 2.5)), float(point + np.percentile(dev, 97.5))


def _diff(pa, ba, pb, bb, rng):
    """Difference (a-b) point, 95% CI, two-sided p, and largest-|effect|-ruled-out (TOST-style).

    Independent bootstrap resamples of each cohort; the two are paired by a fresh random
    permutation (they are independent samples, so index pairing is arbitrary)."""
    if not (len(ba) and len(bb)):
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, bound=np.nan)
    da = pa + (ba - ba.mean())
    db = pb + (bb - bb.mean())
    k = min(len(da), len(db))
    d = rng.permutation(da)[:k] - rng.permutation(db)[:k]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return dict(diff=float(pa - pb), lo=lo, hi=hi, p=p, bound=float(max(abs(lo), abs(hi))))


def _perm_floor(X, yt, rng, B=500):
    """Within-cohort label-permutation floor: same scaffold/class count, shuffled epitopes (M1).
    Returns the 97.5-percentile ARI/AMI achievable by chance on this cohort's own structures."""
    aris, amis = [], []
    y = yt.copy()
    for b in range(B):
        rng.shuffle(y)
        a, m = ari_ami(X, y, seed=b)
        aris.append(a); amis.append(m)
    return float(np.percentile(aris, 97.5)), float(np.percentile(amis, 97.5))


def _fmt_p(p):
    if not np.isfinite(p):
        return "=\\mathrm{n/a}"
    return "<0.001" if p < 1e-3 else ("=%.3f" % p if p < 1e-2 else "=%.2f" % p)


COHORTS = [   # display order: signal calibrators -> system under test -> floor
    ("vdjdb_binder", "VDJdb binder (covered)"),
    ("vdjdb_binder_unc", "VDJdb binder (uncovered)"),   # C1: template-free positive control
    ("immrep22rest", "IMMREP22 rest"),
    ("immrep25", "IMMREP25 pos"),
    ("negatives", "mismatched (floor)"),
]
KEYMAP = {"vdjdb_binder": "Cal", "vdjdb_binder_unc": "Unc", "immrep22rest": "Ii",
          "immrep25": "Imm", "negatives": "Floor"}


def run():
    rng = np.random.default_rng(0)
    rows, boots = [], {}
    for name, label in COHORTS:
        src = "vdjdb_binder" if name == "vdjdb_binder_unc" else name
        df = load_cohort(src)
        if name == "vdjdb_binder":                      # covered real binders (template-contaminated)
            df = df[(df.label == 1) & (df.get("covered", False))].copy()
        elif name == "vdjdb_binder_unc":                # C1: real binders with NO native template
            df = df[(df.label == 1) & (~df.get("covered", False))].copy()
        df = _qualify(df)
        if df.epitope.nunique() < 1 or len(df) < MIN_N:
            print("%-22s  SKIP (insufficient after filter)" % name); continue
        X = _feature_matrix(df)
        Xc = _feature_matrix(df, cdr3_only=True)         # scaffold-free leak-test features (M3)
        yt = pd.Categorical(df.epitope).codes.astype(np.int64)
        s = sn(df, X); s_cdr3 = sn(df, Xc)
        ari, ami = ari_ami(X, yt)
        ab, mb = _boot_arrays(X, yt, rng)
        boots[name] = dict(ari=ab, ami=mb, ari0=ari, ami0=ami)
        mib = mi_report(yt, _partition(X), n_perm=N_PERM)   # MI(epitope; structure-cluster) in bits
        fa, fm = _perm_floor(X, yt, rng)                    # within-cohort permutation floor
        al, ah = _ci(ari, ab); ml, mh = _ci(ami, mb)
        rows.append({"cohort": name, "label": label, **s, "ari": ari, "ami": ami,
                     "ari_lo": al, "ari_hi": ah, "ami_lo": ml, "ami_hi": mh,
                     "mi_bits": mib["excess"], "mi_U": mib["U"], "mi_p": mib["p"],
                     "perm_ari97": fa, "perm_ami97": fm, "sn_cdr3": s_cdr3["sn"],
                     "hyd_mean": float(df[HYD].mean()) if HYD in df else np.nan,
                     "dropped_iptm": int(df.attrs.get("dropped_iptm", 0))})
        print("%-22s ARI=%.3f[%.3f,%.3f] AMI=%.3f[%.3f,%.3f] MI=%.3f bits(U=%.3f,p%s) "
              "permFloorARI=%.3f  SN=%.2f(cdr3 %.2f) n=%d ep=%d hyd=%.2f"
              % (name, ari, al, ah, ami, ml, mh, mib["excess"], mib["U"], _fmt_p(mib["p"]),
                 fa, s["sn"], s_cdr3["sn"], s["n"], s["n_ep"], rows[-1]["hyd_mean"]))
    out = pd.DataFrame(rows)

    # difference tests: system vs floor (expect n.s.), calibrators vs floor/system.
    # The TOST-style equivalence bound each test used to emit is no longer reported -- with these
    # cohort sizes it was too wide to state anything the bootstrap CIs do not already show.
    tests = {"ImmFloor": ("immrep25", "negatives"),
             "CalFloor": ("vdjdb_binder", "negatives"),
             "UncFloor": ("vdjdb_binder_unc", "negatives"),   # C1 decisive
             "CalUnc":   ("vdjdb_binder", "vdjdb_binder_unc"),
             "CalImm":   ("vdjdb_binder", "immrep25"),
             "UncImm":   ("vdjdb_binder_unc", "immrep25"),
             "IiFloor":  ("immrep22rest", "negatives")}
    macros = {}
    for tag, (a, b) in tests.items():
        for metric in ("ari", "ami"):
            A, B = boots.get(a), boots.get(b)
            if A and B:
                d = _diff(A[metric + "0"], A[metric], B[metric + "0"], B[metric], rng)
                macros["structP%s%s" % (metric.capitalize(), tag)] = _fmt_p(d["p"])
    for _, r in out.iterrows():
        k = KEYMAP[r.cohort]
        macros["structAri" + k] = "%.3f" % r.ari
        macros["structAriLo" + k] = "%.3f" % r.ari_lo
        macros["structAriHi" + k] = "%.3f" % r.ari_hi
        macros["structAmi" + k] = "%.3f" % r.ami
        macros["structAmiLo" + k] = "%.3f" % r.ami_lo
        macros["structAmiHi" + k] = "%.3f" % r.ami_hi
        macros["structMi" + k] = "%.3f" % r.mi_bits
        macros["structSn" + k] = "%.2f" % r.sn             # FULL-set diagnostic (scaffold-inflated)
        macros["structSnCdrOnly" + k] = "%.2f" % r.sn_cdr3    # strictly CDR3-only leak test
        macros["structHyd" + k] = "%.2f" % r.hyd_mean
    macros["structIptmMin"] = "%.1f" % IPTM_MIN
    macros["structMinN"] = "%d" % MIN_N
    macros["structNimm"] = "%d" % int(out.loc[out.cohort == "immrep25", "n"].iloc[0])
    macros["structNunc"] = "%d" % int(out.loc[out.cohort == "vdjdb_binder_unc", "n"].iloc[0])

    out.to_csv(os.path.join(RESULTS, "struct_convergence.csv"), index=False)
    xorder = {n: i for i, (n, _) in enumerate(COHORTS)}
    with open(os.path.join(ADAT, "struct_convergence.dat"), "w") as fh:
        fh.write("# x cohort ari ari_lo ari_hi ami ami_lo ami_hi sn sn_cdr3 hyd\n")
        for _, r in out.iterrows():
            fh.write("%d %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
                     % (xorder[r.cohort], r.cohort, r.ari, r.ari_lo, r.ari_hi, r.ami,
                        r.ami_lo, r.ami_hi, r.sn, r.sn_cdr3, r.hyd_mean))
    with open(os.path.join(ADAT, "struct_convergence_macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    fl = out[out.cohort == "negatives"]
    print("\nLEAK TEST (floor must be ~1): FULL SN=%s  CDR3-only SN=%s" %
          (("%.2f" % fl.sn.iloc[0]) if len(fl) else "n/a",
           ("%.2f" % fl.sn_cdr3.iloc[0]) if len(fl) else "n/a"))
    print("wrote results/struct_convergence.csv + struct_convergence.dat + macros")


if __name__ == "__main__":
    run()
