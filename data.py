import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

import torch.serialization as _ts
try:
    import torch_geometric.data.data as _pyg_dd
    import torch_geometric.data.storage as _pyg_ds
    _safe = []
    for _mod, _names in (
        (_pyg_dd, ("Data", "DataEdgeAttr", "DataTensorAttr")),
        (_pyg_ds, ("BaseStorage", "GlobalStorage", "NodeStorage", "EdgeStorage")),
    ):
        for _name in _names:
            _cls = getattr(_mod, _name, None)
            if _cls is not None:
                _safe.append(_cls)
    if _safe:
        _ts.add_safe_globals(_safe)
except ImportError:
    pass

from utils import mask_to_index


def _to_2d_edge_index(ei):
    if torch.is_tensor(ei) and ei.dim() == 2:
        return ei
    if isinstance(ei, (tuple, list)) and len(ei) >= 1:
        first = ei[0]
        if torch.is_tensor(first) and first.dim() == 2:
            return first
        if len(ei) == 2:
            a = ei[0] if torch.is_tensor(ei[0]) else torch.as_tensor(ei[0])
            b = ei[1] if torch.is_tensor(ei[1]) else torch.as_tensor(ei[1])
            return torch.stack([a, b], dim=0)
    raise TypeError(
        f'Cannot normalize edge_index of type {type(ei).__name__}'
    )


class GraphData:

    def __init__(self, features, adj, labels, idx_train, idx_val, idx_test,
                 is_inductive=False, inductive=None):
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.is_inductive = is_inductive
        self.inductive = inductive


def get_dataset(name, data_root=None):
    sagc_root = osp.dirname(osp.realpath(__file__))
    if name in ('ogbn-arxiv', 'ogbn-products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        ogb_root = data_root or osp.join(sagc_root, 'dataset')
        os.makedirs(ogb_root, exist_ok=True)
        dataset = PygNodePropPredDataset(name=name, root=ogb_root)
    elif name in ('reddit', 'reddit2'):
        if name == 'reddit':
            from torch_geometric.datasets import Reddit
            inductive_root = data_root or osp.join(sagc_root, 'data', 'Reddit')
            os.makedirs(inductive_root, exist_ok=True)
            dataset = Reddit(inductive_root)
        else:
            from torch_geometric.datasets import Reddit2
            inductive_root = data_root or osp.join(sagc_root, 'data', 'Reddit2')
            os.makedirs(inductive_root, exist_ok=True)
            dataset = Reddit2(inductive_root)
    else:
        raise NotImplementedError(
            f'Dataset {name!r} not supported. Supported: '
            f'ogbn-arxiv, ogbn-products, reddit, reddit2.'
        )

    splits = None
    if name in ('ogbn-arxiv', 'ogbn-products'):
        splits = dataset.get_idx_split()

    pyg_data = dataset[0]
    n = pyg_data.num_nodes
    if n is None:
        n = int(pyg_data.x.shape[0])
    ei = _to_2d_edge_index(pyg_data.edge_index)

    if name == 'ogbn-arxiv':
        from torch_geometric.utils import to_undirected
        result = to_undirected(edge_index=ei, edge_attr=None,
                               num_nodes=n)
        ei = _to_2d_edge_index(result)
    adj = sp.csr_matrix(
        (np.ones(ei.shape[1]), (ei[0].numpy(), ei[1].numpy())),
        shape=(n, n),
    )

    features = pyg_data.x.numpy()
    labels = pyg_data.y.numpy()
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)

    if hasattr(pyg_data, 'train_mask') and pyg_data.train_mask is not None:
        idx_train = mask_to_index(pyg_data.train_mask, n)
        idx_val = mask_to_index(pyg_data.val_mask, n)
        idx_test = mask_to_index(pyg_data.test_mask, n)
    elif splits is not None:
        idx_train = splits['train'].numpy() if torch.is_tensor(splits['train']) else np.array(splits['train'])
        idx_val = splits['valid'].numpy() if torch.is_tensor(splits['valid']) else np.array(splits['valid'])
        idx_test = splits['test'].numpy() if torch.is_tensor(splits['test']) else np.array(splits['test'])
    else:
        raise RuntimeError(
            f'Dataset {name!r} has neither train_mask nor OGB splits.'
        )

    if name in ('ogbn-arxiv', 'reddit2'):
        scaler = StandardScaler()
        scaler.fit(features[idx_train])
        features = scaler.transform(features)

    data = GraphData(features, adj, labels, idx_train, idx_val, idx_test,
                     is_inductive=name in ('reddit', 'reddit2'))
    if data.is_inductive:
        from inductive import build_inductive_splits
        data.inductive = build_inductive_splits(data)

    return data
