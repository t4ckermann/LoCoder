#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"

if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

echo "Installing LoCoder (dev)..."
"$VENV/bin/pip" install -q -e ".[dev]"

echo ""
echo "Done. Activate your environment with:"
echo "  source .venv/bin/activate"
