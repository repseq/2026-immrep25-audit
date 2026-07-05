#!/usr/bin/env bash
# Fetch the MLR (mixed-lymphocyte-reaction) proliferating-clone samples used for the
# beta-only "expanded clone" control that mimics immrep25's MIRA methodology (TCRB-only,
# expanded/proliferating T cells). 3 reactions x 2 duplicate samples x 2 replicas = 12
# _Proliferating files. Large + gitignored; only the top-200 TCRB/file is used.
# Source: https://huggingface.co/datasets/isalgo/airr_benchmark/tree/main/alice/mlr
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/cache/airr_mlr"; mkdir -p "$DEST"
BASE="https://huggingface.co/datasets/isalgo/airr_benchmark/resolve/main/alice/mlr"
for s in MLR7_TCR1 TCR4 MLR8_TCR2 TCR5 MLR9_TCR3 TCR6; do
  for r in 1 2; do
    out="$DEST/${s}_Proliferating_${r}.tsv.gz"
    [ -f "$out" ] && { echo "have $out"; continue; }
    echo "Downloading ${s}_Proliferating_${r} ..."
    curl -L --fail -o "$out" "$BASE/${s}_Proliferating_${r}.tsv.gz"
  done
done
echo "Done. Build the cohort with: python src/build_mlr.py"
