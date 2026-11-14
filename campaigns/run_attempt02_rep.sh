#!/usr/bin/env bash
set -euo pipefail

replicate="$1"
gpu="$2"
chains="$3"
out="$4"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2
fi

mkdir -p "$(dirname "$out")"

for dataset in wdbc spambase adult; do
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/real_logistic_gpu.py \
    --dataset "$dataset" \
    --replicate "$replicate" \
    --chains "$chains" \
    --priors 0.1 0.3 \
    --methods iid zero_sum antithetic \
    --out "$out"
done
