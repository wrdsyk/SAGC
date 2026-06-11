import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Dict, List, Optional, Union

from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class MLP(nn.Module):

    def __init__(
        self,
        channel_list: List[int],
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        if num_layers is not None:
            assert num_layers == len(channel_list) - 1

        num_layers = len(channel_list) - 1
        self.num_layers = num_layers

        if isinstance(dropout, float):
            dropout = [dropout] * num_layers
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * num_layers

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        self.lins = nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(
                nn.Linear(channel_list[i], channel_list[i + 1], bias=bias[i])
            )

        self.norms = nn.ModuleList()
        norm_count = num_layers - 1 if plain_last else num_layers
        for i in range(norm_count):
            if norm is not None:
                self.norms.append(
                    normalization_resolver(
                        norm, channel_list[i + 1], **(norm_kwargs or {})
                    )
                )
            else:
                self.norms.append(nn.Identity())

    def initialize(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor, return_emb=None) -> Tensor:
        emb = None
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        if isinstance(return_emb, bool):
            x = (x, emb)

        return F.log_softmax(x, dim=1) if not isinstance(x, tuple) else (
            F.log_softmax(x[0], dim=1), x[1]
        )

    @torch.no_grad()
    def predict(self, x: Tensor, return_emb=None) -> Tensor:
        self.eval()
        return self.forward(x, return_emb=return_emb)

    @torch.no_grad()
    def inference(self, x_all: Tensor, batch_size: int,
                  return_emb=None, output_device=None) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        if output_device is None:
            output_device = device
        xs = []
        n = x_all.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = x_all[start:end].to(device)
            for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
                x = lin(x)
                if self.act is not None and self.act_first:
                    x = self.act(x)
                x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = F.dropout(x, p=self.dropout[i], training=self.training)
                emb = x
            if self.plain_last:
                x = self.lins[-1](x)
                x = F.dropout(x, p=self.dropout[-1], training=self.training)
            if isinstance(return_emb, bool):
                x = (x, emb)
            xs.append(x.cpu() if not isinstance(x, tuple) else x)

        x_all = torch.cat(
            [item if not isinstance(item, tuple) else item[0] for item in xs],
            dim=0,
        )
        if str(output_device).startswith('cpu'):
            return F.log_softmax(x_all, dim=1)
        return F.log_softmax(x_all.to(output_device), dim=1)
