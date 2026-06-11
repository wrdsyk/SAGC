#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/run_arxiv.sh
bash scripts/run_products.sh
bash scripts/run_reddit.sh
bash scripts/run_reddit2.sh

python scripts/aggregate.py
