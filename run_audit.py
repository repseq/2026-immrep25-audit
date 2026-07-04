"""Orchestrator: run the full immrep25 audit and write results/ + appendix .dat tables.

Methodology: one-vs-many per epitope (each epitope with >= MIN_N records scored
against all others), aggregated to a dataset geometric mean +/- 95% CI across
epitopes -- no bootstrap resampling. Two OLGA controls (random + pgen-matched)
anchor the noise floor.

Usage: python run_audit.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cohorts import build_cohorts, HIERARCHY, COHORT_META, counts_table
from homology import per_epitope_sn, dataset_sn
from homology_graph import build_graph, graph_stats
from pairing import pairing_per_epitope, dataset_pairing
from publicity import annotate_publicity
from load_data import load_vdjdb, _pair_to_long

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, "results"); os.makedirs(RESULTS, exist_ok=True)
ADAT = os.path.join(REPO, "appendix", "analysis"); os.makedirs(ADAT, exist_ok=True)

MIN_N = 30       # minimum records per epitope for one-vs-many
MAX_D = 3
COLIDX = {n: i for i, n in enumerate(HIERARCHY)}


def _write_dat(df: pd.DataFrame, name: str):
    with open(os.path.join(ADAT, name), "w") as fh:
        fh.write("# " + " ".join(df.columns) + "\n")
        df.to_csv(fh, sep=" ", index=False, header=False)


def _swarm(y, binwidth=0.18, dx=0.035, maxw=0.34):
    """Beeswarm x-offsets: bin points by y and spread each bin symmetrically (no y-jitter)."""
    from collections import defaultdict
    y = np.asarray(y, float)
    off = np.zeros(len(y))
    groups = defaultdict(list)
    for i in np.argsort(y):
        groups[int(round(y[i] / binwidth))].append(i)
    for idxs in groups.values():
        k = len(idxs)
        pos = np.clip((np.arange(k) - (k - 1) / 2.0) * dx, -maxw, maxw)
        for i, p in zip(idxs, pos):
            off[i] = p
    return off


def run():
    coh = build_cohorts(include_olga=True)
    order = [c for c in HIERARCHY if c in coh]
    rng = np.random.default_rng(0)

    ct = counts_table(coh); ct.to_csv(os.path.join(RESULTS, "counts.csv"), index=False)
    print(ct.to_string(index=False)); print()

    # ---------------- homology (one-vs-many, per chain, d=1..3) ----------------
    hom_rows, per_ep = [], {}
    for name in order:
        for chain in ("A", "B"):
            lc = coh[name]["long"]; lc = lc[lc.chain == chain]
            pe = per_epitope_sn(lc, max_d=MAX_D, min_n=MIN_N, seed=1)
            per_ep[(name, chain)] = pe
            for d in range(1, MAX_D + 1):
                s = dataset_sn(pe, d)
                hom_rows.append({"cohort": name, "label": COHORT_META[name]["label"],
                                 "chain": chain, "d": d, **s})
    hom = pd.DataFrame(hom_rows)
    hom.to_csv(os.path.join(RESULTS, "homology_sn.csv"), index=False)
    for chain in ("A", "B"):
        d1 = hom[(hom.chain == chain) & (hom.d == 1)].set_index("cohort").loc[order].reset_index()
        d1 = d1.assign(x=[COLIDX[c] for c in d1.cohort])[["x", "cohort", "sn", "lo", "hi"]]
        _write_dat(d1, f"homology_d1_TR{chain}.dat")
        # d-lines: rows = d, one column per cohort
        wide = {"d": list(range(1, MAX_D + 1))}
        for name in order:
            sub = hom[(hom.cohort == name) & (hom.chain == chain)].sort_values("d")
            wide[name] = sub.sn.values
        _write_dat(pd.DataFrame(wide), f"homology_dlines_TR{chain}.dat")
    print("[homology one-vs-many d1 S/N]")
    print(hom[hom.d == 1][["cohort", "chain", "sn", "lo", "hi", "n_ep"]].to_string(index=False)); print()

    # beeswarm (chain B): per-epitope rate points + boxplot data + non-self reference
    bee, box, ref = [], [], []
    for name in order:
        pe = per_ep[(name, "B")]
        rates = np.maximum(pe.r_self.values, 1e-4) if len(pe) else np.array([])
        xoff = _swarm(np.log10(rates)) if len(rates) else []
        for rate, o in zip(rates, xoff):
            bee.append({"x": COLIDX[name] + o, "rate": rate})
            box.append({"c": COLIDX[name], "rate": rate})
        rn = np.nanmedian(pe.r_nonself.values) if len(pe) else np.nan
        ref.append({"x": COLIDX[name], "r_nonself": max(rn, 1e-4) if rn == rn else 1e-4})
    _write_dat(pd.DataFrame(bee)[["x", "rate"]], "homology_beeswarm.dat")
    _write_dat(pd.DataFrame(box)[["c", "rate"]], "homology_box.dat")
    _write_dat(pd.DataFrame(ref)[["x", "r_nonself"]], "homology_beeswarm_ref.dat")

    # immrep25 by-epitope table (within-epitope Hamming<=1 neighbours per chain)
    peB = per_ep[("immrep25_pos", "B")][["epitope", "n", "m1", "sn1"]].rename(columns={"m1": "mB", "sn1": "snB"})
    peA = per_ep[("immrep25_pos", "A")][["epitope", "m1"]].rename(columns={"m1": "mA"})
    imm_tab = peB.merge(peA, on="epitope").sort_values("snB", ascending=False)
    om = {ch: int(per_ep[("olga_matched", ch)].m1.sum()) if ("olga_matched", ch) in per_ep else 0 for ch in "AB"}
    olga_snB = dataset_sn(per_ep[("olga_matched", "B")], 1)["sn"]
    with open(os.path.join(ADAT, "immrep_epitope_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        fh.write("epitope & $n$ & $\\beta$ nbrs & $\\alpha$ nbrs & $\\beta$ S/N \\\\\n\\midrule\n")
        for _, r in imm_tab.iterrows():
            fh.write("\\texttt{%s} & %d & %d & %d & %.2f \\\\\n" % (r.epitope, r.n, r.mB, r.mA, r.snB))
        fh.write("\\midrule\n\\textbf{immrep25 (geom.\\ mean)} & %d & %d & %d & %.2f \\\\\n" %
                 (int(imm_tab.n.sum()), int(imm_tab.mB.sum()), int(imm_tab.mA.sum()),
                  dataset_sn(per_ep[("immrep25_pos", "B")], 1)["sn"]))
        fh.write("OLGA pgen-matched & %d & %d & %d & %.2f \\\\\n\\bottomrule\n\\end{tabular}\n" %
                 (len(coh["olga_matched"]["paired"]) if "olga_matched" in coh else 0,
                  om["B"], om["A"], olga_snB))
    imm_within = (int(imm_tab.mB.sum()), int(imm_tab.mA.sum()), om["B"], om["A"])

    # ---------------- pairing (one-vs-many, excess bits) ----------------
    pair_rows = []
    for name in order:
        if coh[name]["paired"] is None:
            continue
        pe = pairing_per_epitope(coh[name]["paired"], min_n=MIN_N, n_null=40, seed=1)
        g = dataset_pairing(pe, "gene_excess"); m = dataset_pairing(pe, "mi_excess")
        pair_rows.append({"cohort": name, "label": COHORT_META[name]["label"], "n_ep": g["n_ep"],
                          "gene_mean": g["mean"], "gene_lo": g["lo"], "gene_hi": g["hi"],
                          "mi_mean": m["mean"], "mi_lo": m["lo"], "mi_hi": m["hi"]})
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(os.path.join(RESULTS, "pairing.csv"), index=False)
    pg = pair.assign(x=[COLIDX[c] for c in pair.cohort])[
        ["x", "cohort", "gene_mean", "gene_lo", "gene_hi", "mi_mean", "mi_lo", "mi_hi"]]
    _write_dat(pg, "pairing_excess.dat")
    print("[pairing one-vs-many, excess bits]"); print(pair.to_string(index=False)); print()

    # ---------------- graph stats (chain B, ALL epitopes >= MIN_N records) ----------------
    grows = []
    for name in order:
        lc = coh[name]["long"]; lc = lc[lc.chain == "B"]
        vc = lc.epitope.value_counts(); eps = vc[vc >= MIN_N].index.to_numpy()
        sub = lc[lc.epitope.isin(eps)]                      # all nodes of qualifying epitopes
        if len(sub) > 1000:                                 # cap per cohort (fixed seed), matches figure
            sub = sub.sample(1000, random_state=42)
        sub = sub.reset_index(drop=True)
        st = graph_stats(build_graph(sub)); st["cohort"] = name; st["label"] = COHORT_META[name]["label"]
        st["n_epitopes"] = len(eps); grows.append(st)
    gdf = pd.DataFrame(grows); gdf.to_csv(os.path.join(RESULTS, "graph_stats.csv"), index=False)
    print("[graph stats chain B, all epitopes >=%d]" % MIN_N); print(gdf.to_string(index=False)); print()

    # ---------------- publicity ----------------
    imm = coh["immrep25_pos"]["paired"]
    vlong, _ = load_vdjdb()
    ann = annotate_publicity(imm, vlong, [coh["tcrvdb_true"]["paired"], coh["tcrvdb_false"]["paired"]])
    prows = []
    # non-public subset is small (~12 TCRs/epitope) so use a lower per-epitope floor
    # for BOTH subsets here -- a fair within-immrep with/without-public contrast.
    for label, sub in [("all", ann), ("non_public", ann[~ann.public_any])]:
        lg = _pair_to_long(sub)
        for chain in ("A", "B"):
            pe = per_epitope_sn(lg[lg.chain == chain], max_d=MAX_D, min_n=10, seed=1)
            s = dataset_sn(pe, 1)
            prows.append({"subset": label, "chain": chain, "n": len(sub),
                          "sn": s["sn"], "lo": s["lo"], "hi": s["hi"]})
    pub = pd.DataFrame(prows)
    pub_summary = pd.DataFrame([{"frac_public_a": ann.public_a.mean(), "frac_public_b": ann.public_b.mean(),
                                 "frac_public_any": ann.public_any.mean(),
                                 "frac_public_both": (ann.public_a & ann.public_b).mean()}])
    pub.to_csv(os.path.join(RESULTS, "publicity.csv"), index=False)
    pub_summary.to_csv(os.path.join(RESULTS, "publicity_fractions.csv"), index=False)
    _write_dat(pub.assign(x=range(len(pub)))[["x", "subset", "chain", "sn", "lo", "hi"]], "publicity.dat")
    print("[publicity fractions]"); print(pub_summary.to_string(index=False))
    print("[publicity homology d1 S/N]"); print(pub.to_string(index=False))

    # pgen distributions for the pgen figure
    cache = os.path.join(REPO, "cache", "immrep_pgen.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        for arr, nm in [(d["pa"], "pgen_immrep_A.dat"), (d["pb"], "pgen_immrep_B.dat")]:
            _write_dat(pd.DataFrame({"log10pgen": np.log10(arr[arr > 0])}), nm)
        for ch in ("A", "B"):
            pool = pd.read_csv(os.path.join(REPO, "cache", "olga_pool_%s.tsv" % ch), sep="\t",
                               names=["cdr3", "v", "j", "pgen"])
            pool = pool[pool.pgen > 0].sample(min(20000, len(pool)), random_state=0)
            _write_dat(pd.DataFrame({"log10pgen": np.log10(pool.pgen)}), "pgen_olga_%s.dat" % ch)

    # ---------------- LaTeX macros ----------------
    def H(cohort, chain, d=1):
        r = hom[(hom.cohort == cohort) & (hom.chain == chain) & (hom.d == d)]
        return float(r.sn.iloc[0]) if len(r) else float("nan")

    def P(cohort, col):
        r = pair[pair.cohort == cohort]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    m = {
        "immHomA": H("immrep25_pos", "A"), "immHomB": H("immrep25_pos", "B"),
        "hqHomB": H("vdjdb_hq", "B"), "trueHomB": H("tcrvdb_true", "B"), "lqHomB": H("vdjdb_lq", "B"),
        "falseHomB": H("tcrvdb_false", "B"),
        "olgaMHomB": H("olga_matched", "B"), "olgaRHomB": H("olga_random", "B"),
        "olgaMHomA": H("olga_matched", "A"), "olgaRHomA": H("olga_random", "A"),
        "airrHomB": H("airr_control", "B"), "airrHomA": H("airr_control", "A"),
        "immGene": P("immrep25_pos", "gene_mean"), "immMi": P("immrep25_pos", "mi_mean"),
        "hqGene": P("vdjdb_hq", "gene_mean"), "trueMi": P("tcrvdb_true", "mi_mean"),
        "olgaMGene": P("olga_matched", "gene_mean"), "olgaMMi": P("olga_matched", "mi_mean"),
        "pubAny": 100 * float(pub_summary.frac_public_any.iloc[0]),
        "pubA": 100 * float(pub_summary.frac_public_a.iloc[0]),
        "pubB": 100 * float(pub_summary.frac_public_b.iloc[0]),
        "npHomA": float(pub[(pub.subset == "non_public") & (pub.chain == "A")].sn.iloc[0]),
        "npHomB": float(pub[(pub.subset == "non_public") & (pub.chain == "B")].sn.iloc[0]),
    }
    ints = {"immWithinB": imm_within[0], "immWithinA": imm_within[1],
            "olgaWithinB": imm_within[2], "olgaWithinA": imm_within[3]}
    def fmt(v):
        if not np.isfinite(v):
            return "n/a"
        a = abs(v)
        if a >= 100:
            return "%.0f" % v
        if a >= 10:
            return "%.1f" % v
        if a >= 1:
            return "%.2f" % v
        return "%.3f" % v
    with open(os.path.join(ADAT, "audit_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, fmt(v)))
        for k, v in ints.items():
            fh.write("\\newcommand{\\%s}{%d}\n" % (k, v))
    print("\nDone -> results/ and appendix/analysis/*.dat (+ macros, tables)")


if __name__ == "__main__":
    run()
