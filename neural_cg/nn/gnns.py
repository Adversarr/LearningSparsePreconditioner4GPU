from typing import Union
import torch
from torch import nn

from .basic_layers import FeedForward, MPLayer
from ..utils.weight_init import weight_init


class NodeEdgeProcessing(nn.Module):
    def __init__(
        self,
        node_in_features: int,
        node_out_features: Union[int, None],
        node_encoder: dict,
        node_decoder: dict,
        edge_in_features: int,
        edge_out_features: int,
        edge_encoder: dict,
        edge_decoder: dict,
        num_mp_layers: int,
        # For MPLayer internal
        node_features: int,
        edge_features: int,
        node_residual: bool,
        edge_residual: bool,
        node_mlp: dict,
        edge_mlp: dict,
        msg_mlp: dict,
        msg_norm: bool,
        # For message passing
        aggr: str = "add",
    ):
        super().__init__()
        self.node_enc = FeedForward(
            in_channels=node_in_features,
            out_channels=node_features,
            **node_encoder,
        )

        if node_out_features is None:
            self.node_dec = nn.Identity()
        else:
            self.node_dec = FeedForward(
                in_channels=node_features,
                out_channels=node_out_features,
                **node_decoder,
            )

        self.edge_enc = FeedForward(
            in_channels=edge_in_features,
            out_channels=edge_features,
            **edge_encoder,
        )
        self.edge_dec = FeedForward(
            in_channels=edge_features + 2 * node_features,
            out_channels=edge_out_features,
            **edge_decoder,
        )

        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(
                MPLayer(
                    node_channels=node_features,
                    edge_channels=edge_features,
                    node_residual=node_residual,
                    edge_residual=edge_residual,
                    node_mlp=node_mlp,
                    edge_mlp=edge_mlp,
                    msg_mlp=msg_mlp,
                    aggr=aggr,
                    msg_norm=msg_norm,
                )
            )
        self.apply(weight_init)

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        node_attr = self.node_enc(node_attr)
        edge_attr = self.edge_enc(edge_attr)

        for mp_layer in self.mp_layers:
            node_attr, edge_attr = mp_layer(node_attr, edge_index, edge_attr)
        edge_dec_attr = torch.cat(
            [edge_attr, node_attr[edge_index[0]], node_attr[edge_index[1]]], dim=-1
        )

        node_out: torch.Tensor = self.node_dec(
            node_attr
        )  # Identity do nothing, overhead is small.
        edge_out: torch.Tensor = self.edge_dec(edge_dec_attr)

        return node_out, edge_out
