#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RATES=(0.001 0.005 0.01)
SEEDS=(1 2 3 4 5)

for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== [ogbn-arxiv] reduction_rate=${rate} seed=${seed} ==="
    python main.py \
      --dataset ogbn-arxiv \
      --reduction_rate "${rate}" \
      --seed "${seed}" \
      --feat_prop_k 3 \
      --sap_beta 0.2
  done
done
