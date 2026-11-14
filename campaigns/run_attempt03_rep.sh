#!/usr/bin/env bash
set -euo pipefail

replicate="$1"
gpu="$2"
batch_regime="$3"
out="$4"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2
fi

mkdir -p "$(dirname "$out")"

for dataset in wdbc spambase adult; do
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python campaigns/real_logistic_sgld_gpu.py \
    --dataset "$dataset" \
    --replicate "$replicate" \
    --batch-regime "$batch_regime" \
    --methods iid_iid iid_zero balanced_iid balanced_zero iid_antithetic \
    --out "$out"
done
