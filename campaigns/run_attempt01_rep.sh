#!/usr/bin/env bash
set -euo pipefail

replicate="$1"
gpu="$2"
out="$3"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2
fi

mkdir -p "$(dirname "$out")"

for dataset in wdbc spambase adult; do
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/real_logistic_gpu.py \
    --dataset "$dataset" \
    --replicate "$replicate" \
    --chains 8 \
    --priors 0.1 0.3 \
    --methods zero_sum antithetic \
    --out "$out"

  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/real_logistic_gpu.py \
    --dataset "$dataset" \
    --replicate "$replicate" \
    --chains 128 \
    --priors 0.1 0.3 \
    --methods iid \
    --out "$out"
done
