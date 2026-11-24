#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv or load it on Scholar, then rerun setup.sh." >&2
    exit 1
fi

case "${1:-local}" in
    local) groups=() ;;
    gpu) groups=(--group gpu) ;;
    *) echo "Usage: ./setup.sh [local|gpu]" >&2; exit 2 ;;
esac

if [[ -f uv.lock ]]; then
    uv sync --locked "${groups[@]}"
else
    uv sync "${groups[@]}"
fi
