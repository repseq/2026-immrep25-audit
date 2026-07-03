"""Orchestrator: run the full immrep25 audit and write results/ + appendix .dat tables.

Usage:
    python run_audit.py [--quick]

Produces (results/):
    counts.csv, homology_sn.csv, pairing.csv, publicity.csv, graph_stats.csv
and gnuplot-ready copies in appendix/analysis/*.dat.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META, counts_table
from homology import homology_sn_bootstrap, summarize, per_epitope_sn
from homology_graph import build_graph, graph_stats
from pairing import pairing_bootstrap
from publicity import annotate_publicity
from load_data import load_vdjdb, _pair_to_long, PAIR_COLS

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, "results"); os.makedirs(RESULTS, exist_ok=True)
ADAT = os.path.join(REPO, "appendix", "analysis"); os.makedirs(ADAT, exist_ok=True)

# common matched design (TCRvdb caps the shared epitope count at 2)
K, E_CAP = 30, 2


def _write_dat(df: pd.DataFrame, name: str):
    """Write a whitespace-delimited .dat with a commented header for gnuplot."""
    path = os.path.join(ADAT, name)
    with open(path, "w") as fh:
        fh.write("# " + " ".join(df.columns) + "\n")
        df.to_csv(fh, sep=" ", index=False, header=False)


def load_olga_cohort():
    p = os.path.join(RESULTS, "olga_control.tsv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t")


def run(quick: bool):
    nb_h = 60 if quick else 200
    nb_p, np_p = (15, 60) if quick else (40, 200)

    olga = load_olga_cohort()
    coh = build_cohorts(include_olga=olga is not None, olga_paired=olga)
    order = [c for c in HIERARCHY if c in coh]

    # counts
    ct = counts_table(coh); ct.to_csv(os.path.join(RESULTS, "counts.csv"), index=False)
    print(ct.to_string(index=False)); print()

    # homology (per chain)
    hom_rows = []
    for name in order:
        for chain in ("A", "B"):
            lc = coh[name]["long"]; lc = lc[lc.chain == chain]
            boot = homology_sn_bootstrap(lc, K=K, E_cap=E_CAP, n_boot=nb_h, max_d=3, seed=1)
            s = summarize(boot); s.insert(0, "chain", chain); s.insert(0, "cohort", name)
            s["label"] = COHORT_META[name]["label"]
            hom_rows.append(s)
    hom = pd.concat(hom_rows, ignore_index=True)
    hom.to_csv(os.path.join(RESULTS, "homology_sn.csv"), index=False)
    # gnuplot: one row per cohort with d1 S/N + CI, ordered, per chain, with an x index
    for chain in ("A", "B"):
        d1 = hom[(hom.chain == chain) & (hom.d == 1)].set_index("cohort").loc[order].reset_index()
        d1 = d1.assign(x=range(len(d1)))[["x", "cohort", "sn_median", "sn_lo", "sn_hi"]]
        _write_dat(d1, f"homology_d1_TR{chain}.dat")
    print("[homology d1 S/N]")
    print(hom[hom.d == 1][["cohort", "chain", "sn_median", "sn_lo", "sn_hi"]].to_string(index=False)); print()

    # pairing
    pair_rows = []
    for name in order:
        if coh[name]["paired"] is None:
            continue
        boot = pairing_bootstrap(coh[name]["paired"], K=K, E_cap=E_CAP,
                                 n_boot=nb_p, n_perm=np_p, seed=1)
        q = boot.quantile([0.025, 0.5, 0.975])
        pair_rows.append({"cohort": name, "label": COHORT_META[name]["label"],
                          "gene_bias_z": q.loc[0.5, "gene_bias_z"],
                          "gene_bias_z_lo": q.loc[0.025, "gene_bias_z"],
                          "gene_bias_z_hi": q.loc[0.975, "gene_bias_z"],
                          "mi_z": q.loc[0.5, "mi_z"],
                          "mi_z_lo": q.loc[0.025, "mi_z"],
                          "mi_z_hi": q.loc[0.975, "mi_z"]})
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(os.path.join(RESULTS, "pairing.csv"), index=False)
    pair_g = pair.assign(x=range(len(pair)))[["x", "cohort", "gene_bias_z",
                                              "gene_bias_z_lo", "gene_bias_z_hi",
                                              "mi_z", "mi_z_lo", "mi_z_hi"]]
    _write_dat(pair_g, "pairing_z.dat")
    print("[pairing]"); print(pair.to_string(index=False)); print()

    # graph stats (chain B, legible sample)
    rng = np.random.default_rng(0); grows = []
    for name in order:
        lc = coh[name]["long"]; lc = lc[lc.chain == "B"]
        vc = lc.epitope.value_counts(); eps = vc[vc >= 20].index.to_numpy()
        if eps.size > E_CAP:
            eps = rng.choice(eps, E_CAP, replace=False)
        parts = [lc[lc.epitope == e].sample(min(60, (lc.epitope == e).sum()), random_state=0) for e in eps]
        sub = pd.concat(parts, ignore_index=True) if len(parts) else lc
        st = graph_stats(build_graph(sub)); st["cohort"] = name; st["label"] = COHORT_META[name]["label"]
        grows.append(st)
    gdf = pd.DataFrame(grows); gdf.to_csv(os.path.join(RESULTS, "graph_stats.csv"), index=False)
    print("[graph stats chain B]"); print(gdf.to_string(index=False)); print()

    # --- per-epitope breakdown (chain B): beeswarm points + reference rates ---
    bee, ref = [], []
    per_ep_B = {}
    for i, name in enumerate(order):
        lc = coh[name]["long"]; lc = lc[lc.chain == "B"]
        pe = per_epitope_sn(lc, d=1, min_n=50, max_e=300, max_rest=4000, seed=1)
        per_ep_B[name] = pe
        pe_show = pe.sample(min(40, len(pe)), random_state=1) if len(pe) else pe
        rn = np.nanmedian(pe.r_nonself.values) if len(pe) else np.nan
        ref.append({"x": i, "r_nonself": max(rn, 1e-4) if rn == rn else 1e-4, "n_ep": len(pe)})
        for _, r in pe_show.iterrows():
            bee.append({"x": i + rng.uniform(-0.28, 0.28), "rate": max(r.r_self, 1e-4)})
    _write_dat(pd.DataFrame(bee)[["x", "rate"]], "homology_beeswarm.dat")
    _write_dat(pd.DataFrame(ref)[["x", "r_nonself"]], "homology_beeswarm_ref.dat")

    # immrep25 by-epitope table (n and within-epitope Hamming<=1 neighbour counts, both chains)
    imm_lc = coh["immrep25_pos"]["long"]
    peB = per_ep_B["immrep25_pos"][["epitope", "n", "m"]].rename(columns={"m": "mB"})
    peA = per_epitope_sn(imm_lc[imm_lc.chain == "A"], d=1, min_n=50, seed=1)[["epitope", "m"]].rename(columns={"m": "mA"})
    imm_tab = peB.merge(peA, on="epitope").sort_values("mB", ascending=False)
    olgaB = int(per_ep_B["olga_random"].m.sum()) if "olga_random" in per_ep_B else 0
    olgaA = int(per_epitope_sn(coh["olga_random"]["long"].query("chain=='A'"), d=1, min_n=50, seed=1).m.sum()) if "olga_random" in coh else 0
    with open(os.path.join(ADAT, "immrep_epitope_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        fh.write("epitope & $n$ & $\\beta$ nbrs & $\\alpha$ nbrs \\\\\n\\midrule\n")
        for _, r in imm_tab.iterrows():
            fh.write("\\texttt{%s} & %d & %d & %d \\\\\n" % (r.epitope, r.n, r.mB, r.mA))
        fh.write("\\midrule\n\\textbf{immrep25 total} & %d & %d & %d \\\\\n" %
                 (int(imm_tab.n.sum()), int(imm_tab.mB.sum()), int(imm_tab.mA.sum())))
        fh.write("OLGA control & %d & %d & %d \\\\\n\\bottomrule\n\\end{tabular}\n" %
                 (len(coh["olga_random"]["paired"]) if "olga_random" in coh else 0, olgaB, olgaA))
    imm_within = (int(imm_tab.mB.sum()), int(imm_tab.mA.sum()), olgaB, olgaA)

    # publicity
    imm = coh["immrep25_pos"]["paired"]
    vlong, _ = load_vdjdb()
    ann = annotate_publicity(imm, vlong, [coh["tcrvdb_true"]["paired"], coh["tcrvdb_false"]["paired"]])
    prows = []
    for label, sub in [("all", ann), ("non_public", ann[~ann.public_any])]:
        lg = _pair_to_long(sub)
        for chain in ("A", "B"):
            boot = homology_sn_bootstrap(lg[lg.chain == chain], K=K, E_cap=E_CAP,
                                         n_boot=nb_h, max_d=3, seed=1)
            r = summarize(boot); r = r[r.d == 1].iloc[0]
            prows.append({"subset": label, "chain": chain, "n": len(sub),
                          "sn_median": r.sn_median, "sn_lo": r.sn_lo, "sn_hi": r.sn_hi})
    pub = pd.DataFrame(prows)
    pub.attrs["frac_public_a"] = float(ann.public_a.mean())
    pub_summary = pd.DataFrame([{"frac_public_a": ann.public_a.mean(),
                                 "frac_public_b": ann.public_b.mean(),
                                 "frac_public_any": ann.public_any.mean(),
                                 "frac_public_both": (ann.public_a & ann.public_b).mean()}])
    pub.to_csv(os.path.join(RESULTS, "publicity.csv"), index=False)
    pub_summary.to_csv(os.path.join(RESULTS, "publicity_fractions.csv"), index=False)
    pub_dat = pub.assign(x=range(len(pub)))[["x", "subset", "chain", "sn_median", "sn_lo", "sn_hi"]]
    _write_dat(pub_dat, "publicity.dat")
    print("[publicity fractions]"); print(pub_summary.to_string(index=False))
    print("[publicity homology d1 S/N]"); print(pub.to_string(index=False))

    # pgen distributions (raw log10 values) for the gnuplot pgen figure
    cache = os.path.join(REPO, "cache", "immrep_pgen.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        for arr, nm in [(d["pa"], "pgen_immrep_A.dat"), (d["pb"], "pgen_immrep_B.dat")]:
            v = np.log10(arr[arr > 0])
            _write_dat(pd.DataFrame({"log10pgen": v}), nm)
        poolb = pd.read_csv(os.path.join(REPO, "cache", "olga_pool_b.tsv"), sep="\t")
        poola = pd.read_csv(os.path.join(REPO, "cache", "olga_pool_a.tsv"), sep="\t")
        _write_dat(pd.DataFrame({"log10pgen": np.log10(poolb[poolb.pgen > 0].pgen)}), "pgen_olga_B.dat")
        _write_dat(pd.DataFrame({"log10pgen": np.log10(poola[poola.pgen > 0].pgen)}), "pgen_olga_A.dat")

    # LaTeX macros for reproducible in-text numbers
    def _hom(cohort, chain):
        r = hom[(hom.cohort == cohort) & (hom.chain == chain) & (hom.d == 1)]
        return float(r.sn_median.iloc[0]) if len(r) else float("nan")
    m = {
        "immHomA": _hom("immrep25_pos", "A"), "immHomB": _hom("immrep25_pos", "B"),
        "hqHomB": _hom("vdjdb_hq", "B"), "trueHomB": _hom("tcrvdb_true", "B"),
        "lqHomB": _hom("vdjdb_lq", "B"),
        "olgaHomA": _hom("olga_random", "A"), "olgaHomB": _hom("olga_random", "B"),
        "immGeneZ": float(pair[pair.cohort == "immrep25_pos"].gene_bias_z.iloc[0]),
        "immMiZ": float(pair[pair.cohort == "immrep25_pos"].mi_z.iloc[0]),
        "hqGeneZ": float(pair[pair.cohort == "vdjdb_hq"].gene_bias_z.iloc[0]),
        "pubAny": 100 * float(pub_summary.frac_public_any.iloc[0]),
        "pubA": 100 * float(pub_summary.frac_public_a.iloc[0]),
        "pubB": 100 * float(pub_summary.frac_public_b.iloc[0]),
        "npHomA": float(pub[(pub.subset == "non_public") & (pub.chain == "A")].sn_median.iloc[0]),
        "npHomB": float(pub[(pub.subset == "non_public") & (pub.chain == "B")].sn_median.iloc[0]),
    }
    ints = {"immWithinB": imm_within[0], "immWithinA": imm_within[1],
            "olgaWithinB": imm_within[2], "olgaWithinA": imm_within[3]}
    with open(os.path.join(ADAT, "audit_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%.2f}\n" % (k, v))
        for k, v in ints.items():
            fh.write("\\newcommand{\\%s}{%d}\n" % (k, v))
    print("\nDone -> results/ and appendix/analysis/*.dat (+ audit_macros.tex)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
