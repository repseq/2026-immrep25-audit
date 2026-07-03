"""Homology graph: nodes = CDR3s, edges = Hamming <= 1 (same length).

The within- vs between-epitope edge-density ratio is the graph-theoretic twin of
the homology S/N; epitope assortativity / modularity are its scalar summaries.
Layout coordinates come from graphviz `sfdp` (CLI) so no pygraphviz is required.
A UMAP embedding of k-mer profiles gives the density view.
"""
from __future__ import annotations
import subprocess
import numpy as np
import pandas as pd
import networkx as nx


def _wildcard_edges(seqs: list[str]) -> list[tuple[int, int]]:
    """All index pairs at Hamming distance exactly 1 (same length) via 1-wildcard hashing."""
    from collections import defaultdict
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        for p in range(len(s)):
            buckets[(len(s), p, s[:p] + "*" + s[p + 1:])].append(i)
    edges = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if seqs[i] != seqs[j]:            # distance exactly 1 (0 excluded)
                    edges.add((min(i, j), max(i, j)))
    return list(edges)


def build_graph(sub: pd.DataFrame) -> nx.Graph:
    """sub has columns cdr3, epitope (one chain, one cohort, already subsampled)."""
    seqs = sub.cdr3.tolist()
    epis = sub.epitope.tolist()
    G = nx.Graph()
    for i, (s, e) in enumerate(zip(seqs, epis)):
        G.add_node(i, cdr3=s, epitope=e)
    G.add_edges_from(_wildcard_edges(seqs))
    return G


def graph_stats(G: nx.Graph) -> dict:
    ne = G.number_of_edges()
    if ne == 0:
        return {"n_nodes": G.number_of_nodes(), "n_edges": 0,
                "assortativity": np.nan, "modularity": np.nan, "within_frac": np.nan}
    within = sum(1 for u, v in G.edges if G.nodes[u]["epitope"] == G.nodes[v]["epitope"])
    epi = {n: G.nodes[n]["epitope"] for n in G.nodes}
    comms: dict = {}
    for n, e in epi.items():
        comms.setdefault(e, set()).add(n)
    try:
        assort = nx.attribute_assortativity_coefficient(G, "epitope")
    except Exception:
        assort = np.nan
    mod = nx.algorithms.community.modularity(G, list(comms.values()))
    return {"n_nodes": G.number_of_nodes(), "n_edges": ne,
            "assortativity": assort, "modularity": mod, "within_frac": within / ne}


def sfdp_layout(G: nx.Graph) -> dict:
    """Return {node: (x,y)} using graphviz sfdp via a DOT round-trip."""
    if G.number_of_nodes() == 0:
        return {}
    dot = ["graph g {", "  node [shape=point];"]
    for n in G.nodes:
        dot.append(f"  {n};")
    for u, v in G.edges:
        dot.append(f"  {u} -- {v};")
    dot.append("}")
    try:
        out = subprocess.run(["sfdp", "-Tplain"], input="\n".join(dot),
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return nx.spring_layout(G, seed=0)
    pos = {}
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == "node":
            pos[int(f[1])] = (float(f[2]), float(f[3]))
    return pos or nx.spring_layout(G, seed=0)


def kmer_umap(sub: pd.DataFrame, k: int = 3, seed: int = 0) -> np.ndarray:
    """2-D UMAP of k-mer count profiles (handles variable CDR3 length)."""
    import umap
    from itertools import product
    alpha = "ACDEFGHIKLMNPQRSTVWY"
    kmers = {"".join(p): i for i, p in enumerate(product(alpha, repeat=k))}
    X = np.zeros((len(sub), len(kmers)), dtype=np.float32)
    for r, s in enumerate(sub.cdr3.tolist()):
        for i in range(len(s) - k + 1):
            j = kmers.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1
    n_neighbors = min(15, max(2, len(sub) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=seed)
    return reducer.fit_transform(X)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from cohorts import build_cohorts, HIERARCHY, COHORT_META
    coh = build_cohorts()
    rng = np.random.default_rng(0)
    print(f"{'cohort':<14}{'assort':>8}{'modularity':>12}{'within_frac':>13}{'edges':>8}")
    for name in HIERARCHY:
        if name not in coh:
            continue
        lc = coh[name]["long"]
        lc = lc[lc.chain == "B"]
        # sample up to 8 epitopes x 60 for a legible graph
        vc = lc.epitope.value_counts(); eps = vc[vc >= 20].index.to_numpy()
        if eps.size > 8:
            eps = rng.choice(eps, 8, replace=False)
        parts = [lc[lc.epitope == e].sample(min(60, (lc.epitope == e).sum()), random_state=0) for e in eps]
        sub = pd.concat(parts, ignore_index=True) if parts else lc
        G = build_graph(sub)
        s = graph_stats(G)
        print(f"{COHORT_META[name]['label']:<14}{s['assortativity']:>8.3f}"
              f"{s['modularity']:>12.3f}{s['within_frac']:>13.3f}{s['n_edges']:>8d}")
