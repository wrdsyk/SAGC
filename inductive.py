from dataclasses import dataclass

import scipy.sparse as sp


@dataclass
class InductiveData:
    adj_train: sp.csr_matrix
    adj_val: sp.csr_matrix
    adj_test: sp.csr_matrix


def build_inductive_splits(data):
    adj = data.adj
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    def _induced(idx):
        return adj[idx][:, idx].tocsr()

    return InductiveData(
        adj_train=_induced(data.idx_train),
        adj_val=_induced(data.idx_val),
        adj_test=_induced(data.idx_test),
    )
