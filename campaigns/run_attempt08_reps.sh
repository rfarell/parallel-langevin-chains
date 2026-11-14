#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
init_scale="$2"
mode="$3"
block="$4"
out_prefix="$5"
shift 5

for replicate in "$@"; do
  campaigns/run_attempt08_wallclock_block.sh \
    "$replicate" \
    "$gpu" \
    "$init_scale" \
    "$mode" \
    "$block" \
    "${out_prefix}${replicate}.jsonl"
done
