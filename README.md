# SAGC

Source-Attributed Structure-Free Graph Condensation. The condensed graph uses an identity adjacency, topology is injected once through K-hop feature propagation, and every synthetic node is a sparse convex combination of same-class source features via sparsemax. Anonymized release for a double-blind submission.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA build of torch 2.7.0 with matching torch_geometric / torch_sparse / torch_scatter wheels. A 32 GB GPU is recommended for ogbn-products. All datasets download automatically on first run (Reddit2 is fetched from Google Drive; copy a populated `data/Reddit2/` into the repository root if that is unreachable).

## Usage

Single run:

```bash
python main.py --dataset ogbn-arxiv --reduction_rate 0.005 --seed 1 --feat_prop_k 3 --sap_beta 0.2
```

Reproduce the main table (4 datasets x 3 rates x 5 seeds):

```bash
bash scripts/run_all.sh
```

Each run writes `results/<dataset>/sagc/r{rate}_s{seed}.json` with the test accuracy of the validation-selected student (`best_test_acc`); `scripts/aggregate.py` prints the aggregated table. All unspecified hyperparameters default to the values used in the paper.

## Expected Results

GCN test accuracy (%), mean ± std over seeds 1-5:

| Dataset | Rate | Accuracy | Dataset | Rate | Accuracy |
|---|---:|---:|---|---:|---:|
| ogbn-arxiv | 0.05% | 66.90 ± 0.42 | Reddit | 0.05% | 92.33 ± 0.37 |
| ogbn-arxiv | 0.25% | 68.29 ± 0.24 | Reddit | 0.10% | 92.68 ± 0.19 |
| ogbn-arxiv | 0.50% | 68.57 ± 0.09 | Reddit | 0.20% | 92.73 ± 0.18 |
| ogbn-products | 0.02% | 70.96 ± 0.47 | Reddit2 | 0.05% | 92.01 ± 0.24 |
| ogbn-products | 0.04% | 72.89 ± 0.44 | Reddit2 | 0.10% | 92.26 ± 0.18 |
| ogbn-products | 0.08% | 73.59 ± 0.38 | Reddit2 | 0.20% | 92.57 ± 0.12 |

Rates are paper-level; the CLI `--reduction_rate` values used by the scripts are `code_rate = paper_rate * N_full / N_train` for the transductive OGB datasets and identical for Reddit/Reddit2.

## License

MIT, see [LICENSE](LICENSE).
