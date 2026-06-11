import argparse


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='ogbn-arxiv',
                   choices=['ogbn-arxiv', 'ogbn-products', 'reddit', 'reddit2'])
    p.add_argument('--reduction_rate', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--result_root', type=str, default=None)
    p.add_argument('--feat_prop_k', type=int, default=2)
    p.add_argument('--sap_beta', type=float, default=0.1)
    p.add_argument('--sap_residual_reg', type=float, default=0.0)
    p.add_argument('--diversity_weight', type=float, default=0.1)
    p.add_argument('--sap_kcenter_mass', type=float, default=0.9)
    p.add_argument('--feat_alpha', type=float, default=100)
    p.add_argument('--condensing_loop', type=int, default=1500)
    p.add_argument('--teacher_model_loop', type=int, default=600)
    p.add_argument('--lr_feat', type=float, default=0.01)
    p.add_argument('--lr_model', type=float, default=0.002)
    p.add_argument('--student_model_loop', type=int, default=3000)
    p.add_argument('--student_val_stage', type=int, default=100)
    p.add_argument('--nlayers', type=int, default=2)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--activation', type=str, default='relu')
    p.add_argument('--batch_size', type=int, default=10000)
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)
