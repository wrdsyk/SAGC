#!/usr/bin/env python
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT = os.path.join(ROOT, 'results')
SEEDS = (1, 2, 3, 4, 5)
EXPECTED_SCHEMA_VERSION = 3

CELLS = [
    ('ogbn-arxiv', [('0.001', '0.05%'),
                    ('0.005', '0.25%'),
                    ('0.01', '0.50%')]),
    ('ogbn-products', [('0.0025', '0.02%'),
                       ('0.005', '0.04%'),
                       ('0.01', '0.08%')]),
    ('reddit', [('0.0005', '0.05%'),
                ('0.001', '0.10%'),
                ('0.002', '0.20%')]),
    ('reddit2', [('0.0005', '0.05%'),
                 ('0.001', '0.10%'),
                 ('0.002', '0.20%')]),
]


def warn(msg):
    print(f'WARNING: {msg}', file=sys.stderr)


def read_accuracy(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f'unreadable result file {path}: {exc}')
        return None

    schema = blob.get('schema_version')
    if schema != EXPECTED_SCHEMA_VERSION:
        warn(f'{path}: schema_version={schema!r} '
             f'(expected {EXPECTED_SCHEMA_VERSION}); reading anyway')

    acc = blob.get('best_test_acc')
    if acc is None:
        warn(f'{path}: no accuracy field found, skipping')
        return None

    acc = float(acc)
    if acc <= 1.0:
        acc *= 100.0
    return acc


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    header = (f'{"Dataset":<14} {"Paper rate":>10} {"Code rate":>10} '
              f'{"Accuracy (%)":>16} {"Seeds":>6}')
    print(header)
    print('-' * len(header))

    n_missing_cells = 0
    for dataset, rates in CELLS:
        for code_rate, paper_rate in rates:
            cell_dir = os.path.join(RESULT_ROOT, dataset, 'sagc')
            accs = []
            for seed in SEEDS:
                path = os.path.join(cell_dir, f'r{code_rate}_s{seed}.json')
                if not os.path.isfile(path):
                    warn(f'missing seed {seed} for {dataset} '
                         f'r={code_rate}: {path} not found')
                    continue
                acc = read_accuracy(path)
                if acc is not None:
                    accs.append(acc)

            if not accs:
                n_missing_cells += 1
                cell = '--'
            else:
                if len(accs) < len(SEEDS):
                    warn(f'{dataset} r={code_rate}: only {len(accs)}/'
                         f'{len(SEEDS)} seeds available')
                mean, std = mean_std(accs)
                cell = f'{mean:.2f} +/- {std:.2f}'

            print(f'{dataset:<14} {paper_rate:>10} {code_rate:>10} '
                  f'{cell:>16} {len(accs):>4}/{len(SEEDS)}')

    if n_missing_cells:
        warn(f'{n_missing_cells} of 12 cells have no results at all')
        sys.exit(1)


if __name__ == '__main__':
    main()
