#!/usr/bin/env bash
set -euo pipefail

replicate="$1"
gpu="$2"
init_scale="$3"
mode="$4"
block="$5"
out="$6"

if command -v module >/dev/null 2>&1; then
  module load intel/19.1.1 impi/19.0.9 python3/3.9.7 cuda/12.2
fi

mkdir -p "$(dirname "$out")"

for dataset in wdbc spambase adult; do
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python campaigns/real_logistic_wallclock_gpu.py \
    --dataset "$dataset" \
    --replicate "$replicate" \
    --init-scale "$init_scale" \
    --mode "$mode" \
    --block "$block" \
    --out "$out"
done
