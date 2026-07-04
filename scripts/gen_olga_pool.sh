#!/usr/bin/env bash
# Generate a large OLGA pool with pgen in parallel (GNU parallel).
# Usage: bash scripts/gen_olga_pool.sh [total_per_chain] [n_jobs]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/opt/homebrew/Caskroom/miniconda/base/envs/dev/bin/python"
TOTAL="${1:-100000}"
JOBS="${2:-8}"
CHUNK=$(( TOTAL / JOBS ))
CACHE="$REPO/cache"; mkdir -p "$CACHE"

for chain in A B; do
  echo "Generating $TOTAL $chain sequences in $JOBS jobs of $CHUNK ..."
  seq 0 $((JOBS-1)) | parallel -j "$JOBS" \
    "$PY $REPO/src/olga_gen_chunk.py $chain $CHUNK {}" \
    > "$CACHE/olga_pool_${chain}.tsv"
  echo "  -> $CACHE/olga_pool_${chain}.tsv ($(wc -l < "$CACHE/olga_pool_${chain}.tsv") rows)"
done
