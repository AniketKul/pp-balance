#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
M=nvidia/Nemotron-H-8B-Base-8K
for ctx in 1024 4096 8000; do
  uv run python scripts/run_model.py "$M" \
      --prompt-tokens "$ctx" --pp-sizes 2 4 8 16 \
      --out "results/nemotron-h-8b-${ctx}.json"
done
uv run python scripts/run_model.py Qwen/Qwen3-8B \
    --prompt-tokens 4096 --pp-sizes 2 4 8 16 \
    --out results/qwen3-8b-4096.json
