#!/usr/bin/env bash
# Ouvre la documentation Sphinx dans le navigateur par défaut.
# Reconstruit si nécessaire.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INDEX="$SCRIPT_DIR/docs/build/html/index.html"

if [ ! -f "$INDEX" ]; then
    echo "Documentation absente — reconstruction en cours…"
    uv run python -m sphinx -b html "$SCRIPT_DIR/docs/source" "$SCRIPT_DIR/docs/build/html" -q
fi

xdg-open "$INDEX"
