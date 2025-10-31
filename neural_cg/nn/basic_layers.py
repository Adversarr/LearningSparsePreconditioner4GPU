from typing import Optional, Union
import torch
from torch import nn
from torch_geometric.nn import MessageNorm, MessagePassing
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_torch_coo_tensor,
    to_edge_index,
)


def get_activation(activation: str):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation} not supported.")


def get_normalization(normalization: str, channels: int):
    normalization = normalization.lower()
    if normalization == "none":
        return nn.Identity()
    elif normalization in ["batch", "batchnorm", "batch_norm"]:
        return nn.RMSNorm(channels)
    elif normalization in ["layer", "layernorm", "layer_norm"]:
        return nn.LayerNorm(channels)
    elif normalization in ["rms", "rmsnorm", "rms_norm"]:
        return nn.RMSNorm(channels)
    else:
        raise ValueError(f"Normalization {normalization} not supported.")


class PositionalEncoding(nn.Module):
    """Computes the Sine positional encoding of a tensor.

    Attributes:
        freqs: Frequencies applied to each dimension
    """

    def __init__(
        self, n_freqs: int = 1, base_freq: float = torch.pi, exp_scaling: bool = True
    ):
        super().__init__()
        freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32) * base_freq
        if exp_scaling:
            freqs = torch.exp2(freqs)
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def estimate_output_ndim(self, input_physical_dim: int):
        return (1 + self.freqs.shape[0]) * input_physical_dim

    def forward(self, x: torch.Tensor):
        physical_dim = x.shape[-1]
        ys = [torch.sin(x[..., [i]] * self.freqs) for i in range(physical_dim)]
        y = torch.cat(ys + [x], dim=-1)
        return y


class FeedForward(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        pre_norm: str = "none",
        activation: str = "gelu",
        out_activation: str = "none",
    ):
        super().__init__()
        self.pre_norm = get_normalization(pre_norm, in_channels)
        self.lift = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            get_activation(activation),
        )
        self.body = nn.ModuleList()
        for _ in range(1, num_layers):
            self.body.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    get_activation(activation),
                )
            )
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            get_activation(out_activation),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.lift(x)
        for layer in self.body:
            x = layer(x)
        x = self.proj(x)
        return x


class GraphSpmv(MessagePassing):
    """Perform Sparse Matrix-Vector multiplication, in a message passing manner.

        Blocked Sparse matrix multiplication:
            $$ y_i = sum_j A_{i,j} * x_j $$
        where:
            - A is a sparse matrix, blocked, A_{i, j} is R^{dim X dim} matrix
            - x_j is a vector, R^{dim}
            - y_i is a vector, R^{dim}

    Attributes:
        transpose: Whether to transpose the matrix.
    """

    def __init__(self, use_transpose=False):
        flow = "target_to_source" if not use_transpose else "source_to_target"
        self.transpose = use_transpose
        super(GraphSpmv, self).__init__(aggr="add", flow=flow)

    def forward(self, X, edge_index, A, mask=None):
        out = self.propagate(edge_index, x=X, edge_attr=A)
        if mask is not None:
            out = out * mask
        return out

    def message(self, x_i, x_j, edge_attr):  # type: ignore
        if self.transpose:
            result = torch.bmm(edge_attr.transpose(-1, -2), x_j.unsqueeze(-1))
        else:
            result = torch.bmm(edge_attr, x_j.unsqueeze(-1))
        return result.squeeze(-1)


class MPLayer(MessagePassing):
    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        node_residual: bool,
        edge_residual: bool,
        node_mlp: dict,
        edge_mlp: dict,
        msg_mlp: dict,
        aggr: str = "add",
        msg_norm: bool = True,
    ):
        """Initialize the Message Passing Layer with Node&Edge features

        Args:
            node_channels: number of node features
            edge_channels: number of edge features
            residual_node: add residual connection
            node_mlp: config of node processing mlp
            edge_mlp: config
            msg_mlp: confi
            aggr: ['mean', 'add']
        """

        super().__init__(aggr=aggr)

        self.node_mlp = FeedForward(
            in_channels=node_channels,
            out_channels=node_channels,
            **node_mlp,
        )
        self.edge_mlp = FeedForward(
            in_channels=2 * node_channels + edge_channels,
            out_channels=edge_channels,
            **edge_mlp,
        )
        self.msg_mlp = FeedForward(
            in_channels=edge_channels + 2 * node_channels,
            out_channels=node_channels,
            **msg_mlp,
        )

        self.node_residual = node_residual
        self.edge_residual = edge_residual
        if msg_norm:
            self.node_msg_norm = MessageNorm()

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        node_attr_new = self.propagate(edge_index, x=node_attr, edge_attr=edge_attr)
        if hasattr(self, "msg_norm"):
            node_attr_new = self.node_msg_norm(node_attr, node_attr_new)

        if self.node_residual:
            node_attr_output = node_attr + node_attr_new
        else:
            node_attr_output = node_attr_new

        edge_attr_new = self.edge_updater(edge_index, x=node_attr, edge_attr=edge_attr)
        if self.edge_residual:
            edge_attr_output = edge_attr + edge_attr_new
        else:
            edge_attr_output = edge_attr_new

        return node_attr_output, edge_attr_output

    def edge_update(self, edge_attr, x_i, x_j) -> torch.Tensor:  # type: ignore
        efeat = [x_i, x_j, edge_attr]
        return self.edge_mlp(torch.cat(efeat, dim=-1))

    def message(self, x_i, x_j, edge_attr) -> torch.Tensor:  # type: ignore
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(features)

    def update(self, aggr_out, x):  # type: ignore
        return self.node_mlp(aggr_out)


class AATPE(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.spmv = GraphSpmv()
        self.spmv_t = GraphSpmv(use_transpose=True)
        self.epsilon = epsilon

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        boo_values,
        mask: Union[torch.Tensor, None] = None,
        diag: Optional[torch.Tensor] = None,
    ):
        """
        If diag is None:
            y <- eps x + A^T @ A @ x
        otherwise:
            y <- eps diag x + A^T @ diag A @ x
        """


        # [nVerts, nBlock] -> [nVerts, nBlock] -> [nVerts, nBlock]
        AT_x = self.spmv_t(X=x, edge_index=edge_index, A=boo_values, mask=mask)
        eps_x: torch.Tensor = self.epsilon * x
        if diag is not None:
            assert diag.shape == AT_x.shape
            AT_x = AT_x * diag
            eps_x = eps_x * diag

        A_AT_x = self.spmv(X=AT_x, edge_index=edge_index, A=boo_values, mask=mask)

        return eps_x + A_AT_x


class LLT(nn.Module):
    def __init__(self):
        super().__init__()
        self.spmv = GraphSpmv()
        self.spmv_t = GraphSpmv(use_transpose=True)

    def forward(
        self, x, edge_index, boo_values, mask: Union[torch.Tensor, None] = None
    ):
        LT_x = self.spmv_t(X=x, edge_index=edge_index, A=boo_values, mask=mask)
        L_LT_x = self.spmv(X=LT_x, edge_index=edge_index, A=boo_values, mask=mask)
        return L_LT_x


class ToLowerTriangular(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, data, edge_index):
        if not self.inplace:
            data = data.clone()

        # transform the data into lower triag graph
        rows, cols = edge_index[0], edge_index[1]
        fil = cols <= rows
        l_index = edge_index[:, fil]
        edge_embedding = data[fil]

        # data.edge_index, data.edge_attr = l_index, edge_embedding
        return edge_embedding, l_index


class TwoHop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, num_nodes, edge_index, edge_attr):
        assert edge_index is not None
        num_nodes_tmp=num_nodes

        edge_index_tmp,edge_attr_tmp=edge_index.clone(),edge_attr.clone()

        adj = to_torch_coo_tensor(edge_index_tmp, size=num_nodes_tmp)

        adj = adj @ adj

        edge_index2, _ = to_edge_index(adj)
        edge_index2, _ = remove_self_loops(edge_index2)

        edge_index_tmp = torch.cat([edge_index_tmp, edge_index2], dim=1)

        if edge_attr_tmp is not None:
            # We treat newly added edge features as "zero-features":
            edge_attr2 = edge_attr_tmp.new_zeros(edge_index2.size(1), *edge_attr_tmp.size()[1:])
            edge_attr_tmp = torch.cat([edge_attr_tmp, edge_attr2], dim=0)

        edge_index_tmp, edge_attr_tmp = coalesce(edge_index_tmp, edge_attr_tmp, num_nodes_tmp)

        return edge_index_tmp, edge_attr_tmp

class ToLowerTriangularAndConsistSparse(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, data, edge_index, drop_fol):
        if not self.inplace:
            data = data.clone()

        # transform the data into lower triag graph
        rows, cols = edge_index[0], edge_index[1]
        # __import__("pdb").set_trace()
        fil = (cols <= rows) & (torch.abs(data.flatten()) >= drop_fol)
        l_index = edge_index[:, fil]
        edge_embedding = data[fil]

        # data.edge_index, data.edge_attr = l_index, edge_embedding
        return edge_embedding, l_index
