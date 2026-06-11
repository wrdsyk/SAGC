#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RATES=(0.0025 0.005 0.01)
SEEDS=(1 2 3 4 5)

for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== [ogbn-products] reduction_rate=${rate} seed=${seed} ==="
    python main.py \
      --dataset ogbn-products \
      --reduction_rate "${rate}" \
      --seed "${seed}"
  done
done
