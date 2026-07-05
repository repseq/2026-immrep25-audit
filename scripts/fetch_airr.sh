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

# SRA per-donor samples (isalgo/airr_benchmark) for the AIRR non-random control
BENCH="$REPO/cache/airr_bench"; mkdir -p "$BENCH"
BB="https://huggingface.co/datasets/isalgo/airr_benchmark/resolve/main/sra"
[ -f "$BENCH/meta.tsv" ] || curl -L --fail -o "$BENCH/meta.tsv" "$BB/meta.tsv"
if [ ! -d "$BENCH/samples" ]; then
  echo "Downloading + unpacking SRA samples ..."
  curl -L --fail -o "$BENCH/samples.tar.gz" "$BB/samples.tar.gz"
  mkdir -p "$BENCH/samples" && tar xzf "$BENCH/samples.tar.gz" -C "$BENCH/samples"
fi
echo "Done. Build the cohorts with: python src/build_airr.py"
