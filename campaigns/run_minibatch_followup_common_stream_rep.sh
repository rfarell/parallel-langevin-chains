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
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python campaigns/real_logistic_sgld_gpu.py \
    --dataset "$dataset" \
    --priors 0.1 0.3 \
    --replicate "$replicate" \
    --batch-regime primary \
    --methods iid_iid iid_zero iid_antithetic \
    --iid-batch-replacement \
    --common-batch-stream \
    --init-mode iid_normal \
    --init-scale 0.25 \
    --out "$out"
done
