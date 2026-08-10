"""Generate one chunk of OLGA sequences with pgen (for GNU parallel).

Usage: python src/olga_gen_chunk.py <A|B> <n> <seed>
Writes TSV rows `cdr3<TAB>v<TAB>j<TAB>pgen` to stdout.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olga_control import generate_pool

if __name__ == "__main__":
    chain, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    df = generate_pool(chain, n, seed=seed)
    df.to_csv(sys.stdout, sep="\t", index=False, header=False)
