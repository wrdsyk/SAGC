import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter
from typing import Any, Callable, Dict, List, Optional, Union

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import zeros


class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = PyGLinear(in_channels, out_channels, bias=False,
                             weight_initializer='glorot')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class BasicGNN(nn.Module):

    supports_edge_weight: bool = True
    supports_edge_attr: bool = False

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        sgc: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.temp_layers = nlayers
        self.nlayers = 1 if sgc else nlayers
        if sgc:
            nlayers = 1
        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm_name = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs
        self.sgc = sgc
        self.nclass = nclass if nclass is not None else nhid

        self.convs = ModuleList()
        in_dim = nfeat
        if nlayers > 1:
            self.convs.append(self.init_conv(in_dim, nhid, **kwargs))
            in_dim = nhid
        for _ in range(nlayers - 2):
            self.convs.append(self.init_conv(in_dim, nhid, **kwargs))
            in_dim = nhid
        if nclass is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(in_dim, self.nclass, **kwargs))
        else:
            self.convs.append(self.init_conv(in_dim, nhid, **kwargs))

        self.norms = None
        if norm is not None and not sgc:
            norm_layer = normalization_resolver(
                norm, nhid, **(norm_kwargs or {}))
            self.norms = ModuleList()
            for _ in range(nlayers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, nhid, nlayers)
        if jk is not None:
            jk_dim = nlayers * nhid if jk == 'cat' else nhid
            self.lin = Linear(jk_dim, self.nclass)

    def init_conv(self, nfeat: int, nclass: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *,
                edge_weight: OptTensor = None,
                edge_attr: OptTensor = None,
                return_embed: bool = False) -> Tensor:
        xs: List[Tensor] = []
        penultimate = None
        for i in range(self.nlayers):
            if i == self.nlayers - 1 and return_embed:
                penultimate = x
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.nlayers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        if return_embed:
            return x, penultimate
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def predict(self, x: Tensor, edge_index: Adj, *,
                edge_weight: OptTensor = None,
                edge_attr: OptTensor = None) -> Tensor:
        self.eval()
        return self.forward(x, edge_index, edge_weight=edge_weight,
                            edge_attr=edge_attr)

    @torch.no_grad()
    def inference(self, x_all: Tensor, loader: NeighborSampler,
                  device: Optional[torch.device] = None,
                  output_device: Optional[torch.device] = None) -> Tensor:
        self.eval()
        if output_device is None:
            output_device = device
        jk_xs: List[Tensor] = []
        for i in range(self.temp_layers):
            xs: List[Tensor] = []
            for batch_size, n_id, adj in loader:
                x = x_all[n_id].to(device)
                edge_index = adj.adj_t.to(device)
                if not self.sgc:
                    x = self.convs[i](x, edge_index)[:batch_size]
                else:
                    x = self.convs[0].propagate(edge_index, x=x)[:batch_size]
                    if i == self.temp_layers - 1:
                        x = self.convs[0].lin(x)
                if not self.sgc:
                    if i == self.nlayers - 1 and self.jk_mode is None:
                        xs.append(x.cpu())
                        continue
                    if self.act is not None and self.act_first:
                        x = self.act(x)
                    if self.norms is not None:
                        x = self.norms[i](x)
                    if self.act is not None and not self.act_first:
                        x = self.act(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
            if hasattr(self, 'jk'):
                jk_xs.append(x_all)
        if hasattr(self, 'jk'):
            jk_dev = [t.to(device) for t in jk_xs]
            x = self.jk(jk_dev)
            if hasattr(self, 'lin'):
                x = self.lin(x)
            x_all = x.cpu()
        return F.log_softmax(x_all, dim=1).to(output_device)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, nlayers={self.nlayers})')


class GCN(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int, **kwargs) -> MessagePassing:
        return GCNConv(nfeat, nclass, **kwargs)
