#!/usr/bin/env bash
# Fetch the human AIRR control repertoire (vdjtools .aa, TRA + TRB) used as the
# real post-thymic-selection floor. Large + gitignored; only a 1000-seq sample is used.
# Source: https://huggingface.co/datasets/isalgo/airr_control
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/cache/airr"; mkdir -p "$DEST"
BASE="https://huggingface.co/datasets/isalgo/airr_control/resolve/main"
for chain in tra trb; do
  out="$DEST/human.${chain}.aa.tsv.gz"
  if [ -f "$out" ]; then echo "have $out"; continue; fi
  echo "Downloading human.${chain}.aa ..."
  curl -L --fail -o "$out" "$BASE/human.${chain}.aa.vdjtools.tsv.gz"
done
echo "Done. Build the cohort with: python src/build_airr.py"
