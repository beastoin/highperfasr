#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HEADER="<!-- AUTO-GENERATED — do not edit. Source: web/deploy/deploy.html -->"

src="$REPO_ROOT/web/deploy/deploy.html"
dst="$REPO_ROOT/docs/deploy.html"

if [[ ! -f "$src" ]]; then
  echo "ERROR: source not found: $src" >&2
  exit 1
fi

{
  echo "$HEADER"
  cat "$src"
} > "$dst"

echo "Synced: web/deploy/deploy.html → docs/deploy.html"
