#!/usr/bin/env bash
# Fetch the public VDJdb release used by this audit (not committed: ~400 MB unpacked).
# Source: https://github.com/antigenomics/vdjdb-db/releases/tag/2026-06-11-ZENODO
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/dump"
URL="https://github.com/antigenomics/vdjdb-db/releases/download/2026-06-11-ZENODO/vdjdb-2026-06-03.zip"
ZIP="$DEST/vdjdb-2026-06-03.zip"

mkdir -p "$DEST"
if [ -d "$DEST/vdjdb-2026-06-03" ] && [ -f "$DEST/vdjdb-2026-06-03/vdjdb.txt" ]; then
  echo "VDJdb already present at $DEST/vdjdb-2026-06-03 — nothing to do."
  exit 0
fi

echo "Downloading VDJdb release (~41 MB zip) ..."
curl -L --fail -o "$ZIP" "$URL"
echo "Unpacking ..."
unzip -o -q "$ZIP" -d "$DEST"
rm -f "$ZIP"
echo "Done: $DEST/vdjdb-2026-06-03/"
ls -1 "$DEST/vdjdb-2026-06-03/" | head
