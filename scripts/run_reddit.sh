#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RATES=(0.0005 0.001 0.002)
SEEDS=(1 2 3 4 5)

for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== [reddit] reduction_rate=${rate} seed=${seed} ==="
    python main.py \
      --dataset reddit \
      --reduction_rate "${rate}" \
      --seed "${seed}" \
      --condensing_loop 2500 \
      --activation sigmoid \
      --lr_model 0.001
  done
done
