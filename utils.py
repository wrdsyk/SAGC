import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    col = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    indices = torch.cat((row, col), 1).t()
    values = torch.FloatTensor(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape))


def to_scipy(tensor):
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix(
            (values.cpu().numpy(), indices.cpu().numpy()),
            shape=tensor.shape,
        )
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix(
            (values.cpu().numpy(), indices.cpu().numpy()),
            shape=tensor.shape,
        )


def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels)
    labels = labels.to(output.device)
    preds = output.max(1)[1].to(dtype=labels.dtype)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


def to_tensor(adj, device='cpu'):
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    return adj.to(device)


def is_sparse_tensor(tensor):
    return tensor.layout == torch.sparse_coo


def normalize_adj(mx):
    if not isinstance(mx, sp.lil_matrix):
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_adj_tensor(adj, sparse=False):
    device = adj.device
    if sparse:
        adj_scipy = to_scipy(adj)
        mx = normalize_adj(adj_scipy)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx @ r_mat_inv
        return mx


def mask_to_index(mask, size):
    all_idx = np.arange(size)
    return all_idx[mask]


def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
             add_self_loops=True):
    from torch_geometric.utils import add_remaining_self_loops
    from torch_scatter import scatter_add

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), device=edge_index.device
        )

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1.0, num_nodes
        )

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
