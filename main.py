import os
import random
import functools
import gc

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

_original_torch_load = torch.load


def _safe_remap(storage, location):
    if isinstance(location, str) and location.startswith('cuda'):
        try:
            idx = int(location.split(':', 1)[1]) if ':' in location else 0
        except (IndexError, ValueError):
            return storage
        if torch.cuda.is_available() and idx >= torch.cuda.device_count():
            return storage.cuda(0)
    return storage


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    if 'map_location' not in kwargs and len(args) < 2:
        kwargs['map_location'] = _safe_remap
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from config import parse_args
from data import get_dataset
from utils import to_tensor, is_sparse_tensor, normalize_adj_tensor, gcn_norm
from models import GCN
from condensation import generate_labels_syn, train_teacher, build_cache_path
from training import train_on_syn_graph
from attributive import (AttributiveFeatSyn, node_condensation_sap,
                         node_condensation_sap_streaming,
                         kcenter_selected_indices_per_class)

_STREAMING_CONSTANTS = (
    ('sap_streaming_strategy', 'balanced'),
    ('sap_streaming_target_gb', 8.0),
    ('sap_streaming_max_chunk_size', 8192),
    ('sap_streaming_cache_real', 1),
    ('sap_streaming_bn_mode', 'calibrated_eval'),
    ('sap_streaming_bn_calib_interval', 5),
    ('sap_streaming_bn_chunk_size', 256),
    ('sap_streaming_class_chunk_size', 256),
    ('sap_streaming_stat_interval', 5),
    ('sap_streaming_precision', 'fp32'),
)


def main(argv=None):
    args = parse_args(argv)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sagc_root = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(sagc_root, 'saved')
    result_dir = args.result_root or os.path.join(sagc_root, 'results')
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(sagc_root, result_dir)
    for subdir in ['temp', 'feat', 'teacher', 'student']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

    data = get_dataset(args.dataset, data_root=args.data_root)
    use_streaming = (not data.is_inductive) or (args.reduction_rate > 0.002)
    if use_streaming:
        for k, v in _STREAMING_CONSTANTS:
            setattr(args, k, v)

    feat = torch.FloatTensor(data.features).detach().cuda()
    adj = to_tensor(data.adj, device='cpu').detach()
    labels = torch.LongTensor(data.labels).cuda()
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if data.is_inductive:
        adj_train_csr = data.inductive.adj_train
        adj_val_csr = data.inductive.adj_val
        adj_test_csr = data.inductive.adj_test
        adj_train_full = to_tensor(adj_train_csr, device='cpu').detach()
        adj_val_full = to_tensor(adj_val_csr, device='cpu').detach()
        adj_test_full = to_tensor(adj_test_csr, device='cpu').detach()
        adj = adj_train_full

    feat_prop_k = args.feat_prop_k
    if feat_prop_k > 0:
        def _normalized_adj(adj_csr):
            adj_aug = (adj_csr + sp.eye(
                adj_csr.shape[0], dtype=np.float32, format='csr')).astype(
                    np.float32)
            rowsum = np.array(adj_aug.sum(1)).flatten().astype(np.float32)
            r_inv = np.zeros_like(rowsum, dtype=np.float32)
            nz = rowsum > 0
            r_inv[nz] = np.power(rowsum[nz], -0.5)
            D_inv = sp.diags(r_inv)
            return (D_inv @ adj_aug @ D_inv).astype(np.float32)

        def _propagate(features_np, adj_csr, k):
            a_norm = _normalized_adj(adj_csr).tocoo()
            a_st = SparseTensor(
                row=torch.from_numpy(a_norm.row).long(),
                col=torch.from_numpy(a_norm.col).long(),
                value=torch.from_numpy(a_norm.data).float(),
                sparse_sizes=(a_norm.shape[0], a_norm.shape[1]),
            ).cuda()
            h = torch.from_numpy(features_np).float().cuda()
            for _ in range(k):
                h = a_st @ h
            del a_st
            return h.detach()

        if data.is_inductive:
            feat_clone = feat.clone()
            feat_clone[idx_train] = _propagate(
                data.features[idx_train], adj_train_csr, feat_prop_k)
            feat_clone[idx_val] = _propagate(
                data.features[idx_val], adj_val_csr, feat_prop_k)
            feat_clone[idx_test] = _propagate(
                data.features[idx_test], adj_test_csr, feat_prop_k)
            feat = feat_clone
        else:
            feat = _propagate(data.features, data.adj, feat_prop_k)
        print(f'[Feature Propagation] K={feat_prop_k}, '
              f'feat shape={tuple(feat.shape)}')

    feat_train = feat[idx_train]
    feat_test = feat[idx_test]
    labels_train = labels[idx_train]
    labels_test = labels[idx_test]
    d = feat.shape[1]
    nclass = int(labels.max() + 1)

    labels_syn, num_class_dict = generate_labels_syn(
        labels_train, feat_train, args, result_dir, device)
    n = len(labels_syn)

    index = []
    index_syn = []
    for c in range(nclass):
        index.append(torch.where(labels_train == c))
        index_syn.append(torch.where(labels_syn == c))

    feat_syn_path = build_cache_path(save_dir, args)
    if not os.path.exists(feat_syn_path):
        print(f'[SAP] condensing {n} synthetic nodes')
        teacher = train_teacher(
            feat_train, labels_train, feat_test, labels_test,
            d, nclass, args, save_dir, device)
        kc_per_class = kcenter_selected_indices_per_class(
            feat_train, labels_train, num_class_dict)
        feat_syn_module = AttributiveFeatSyn(
            labels_syn=labels_syn.cpu(),
            X_orig=feat_train,
            labels_train=labels_train.cpu(),
            device=device,
            beta=args.sap_beta,
        ).to(device)
        feat_syn_module.init_kcenter(
            kc_per_class, kcenter_mass=args.sap_kcenter_mass)
        if use_streaming:
            node_condensation_sap_streaming(
                feat_syn_module, labels_syn, feat_train, labels_train,
                index, index_syn, teacher,
                nclass, d, args, feat_syn_path, device)
        else:
            node_condensation_sap(
                feat_syn_module, labels_syn, feat_train, labels_train,
                index, index_syn, teacher,
                nclass, d, args, feat_syn_path, device)
    feat_syn = torch.load(feat_syn_path, map_location='cpu').detach().cuda()

    n_syn = feat_syn.shape[0]
    loop = torch.arange(n_syn, device=device)
    edge_index_syn = torch.stack([loop, loop], dim=0).long()
    edge_weight_syn = torch.ones(n_syn, device=device)

    def _prepare_eval_adj(adj_t, cache_tag):
        cache_path = os.path.join(save_dir, 'temp', f'adj_norm_{cache_tag}.pt')
        if os.path.exists(cache_path):
            adj_norm = torch.load(cache_path).detach()
        else:
            adj_norm = (normalize_adj_tensor(adj_t, sparse=True)
                        if is_sparse_tensor(adj_t)
                        else normalize_adj_tensor(adj_t))
            torch.save(adj_norm, cache_path)
        return SparseTensor(
            row=adj_norm._indices()[0], col=adj_norm._indices()[1],
            value=adj_norm._values(), sparse_sizes=adj_norm.size()).t()

    if data.is_inductive:
        adj_train_norm = _prepare_eval_adj(
            adj_train_full, f'{args.dataset}_train_{args.seed}')
        adj_val_norm = _prepare_eval_adj(
            adj_val_full, f'{args.dataset}_val_{args.seed}')
        adj_test_norm = _prepare_eval_adj(
            adj_test_full, f'{args.dataset}_test_{args.seed}')
        adj = adj_train_norm
    else:
        adj = _prepare_eval_adj(adj, f'{args.dataset}_{args.seed}')

    edge_index_syn, edge_weight_syn = gcn_norm(
        edge_index_syn, edge_weight_syn, n, add_self_loops=False)

    try:
        del feat_syn_module, teacher
    except NameError:
        pass
    del feat_train, feat_test, labels_train, labels_test, index, index_syn
    gc.collect()
    torch.cuda.empty_cache()

    model = GCN(nfeat=d, nhid=args.hidden, nclass=nclass,
                dropout=args.dropout, nlayers=args.nlayers,
                norm='BatchNorm', act=args.activation).cuda()
    model.initialize()

    inductive_eval = None
    if data.is_inductive:
        inductive_eval = {
            'feat_train': feat[idx_train],
            'adj_train': adj_train_norm,
            'feat_val': feat[idx_val],
            'adj_val': adj_val_norm,
            'feat_test': feat[idx_test],
            'adj_test': adj_test_norm,
        }

    best_test = train_on_syn_graph(
        model, feat_syn, edge_index_syn, edge_weight_syn,
        labels_syn, feat, adj, labels,
        idx_train, idx_val, idx_test, args,
        save_dir, result_dir, device,
        inductive_eval=inductive_eval)

    return best_test


if __name__ == '__main__':
    main()
