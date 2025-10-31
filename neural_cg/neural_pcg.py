from copy import deepcopy
import math

import torch
from torch import nn
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import segregate_self_loops, sort_edge_index
import torch_scatter
import torch_scatter


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type=None,
        last_layer_nonlinearity=None,
    ):
        """
        MLP
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer
        hidden_layers: number of hidden layers
        ## TODO: maybe need to add normalization layers
        """

        super(MLP, self).__init__()

        activation = nn.ReLU
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        if last_layer_nonlinearity is not None:
            layers.append(last_layer_nonlinearity())

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "InstanceNorm1d",
                "LazyInstanceNorm1d",
                "BatchNorm1d",
                "LazyBatchNorm1d",
                "MessageNorm",
            ]
            if norm_type in [
                "LayerNorm",
                "BatchNorm1d",
                "InstanceNorm",
                "LazyInstanceNorm1d",
                "LazyBatchNorm1d",
            ]:
                norm_layer = getattr(nn, norm_type)
            else:
                raise NotImplementedError
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPTanh(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type=None,
        last_layer_nonlinearity=None,
    ):
        super(MLPTanh, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        if last_layer_nonlinearity is not None:
            layers.append(last_layer_nonlinearity())

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "InstanceNorm1d",
                "LazyInstanceNorm1d",
                "BatchNorm1d",
                "LazyBatchNorm1d",
                "MessageNorm",
            ]
            if norm_type in [
                "LayerNorm",
                "BatchNorm1d",
                "InstanceNorm",
                "LazyInstanceNorm1d",
                "LazyBatchNorm1d",
            ]:
                norm_layer = getattr(nn, norm_type)
            else:
                raise NotImplementedError
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EdgeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):
        """
        Edge processor: propogate node features, and applies edge functions

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        """

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge,
            in_dim_edge,
            hidden_dim,
            hidden_layers,
            norm_type,
        )

    def forward(self, node_feature, edge_matrix, edge_feature):
        """
        node_feature: node feature: B x fdim
        edge:  [sender, receiver] edge_matrix, B x 2
        edge feature: B x fdim
        """
        # concatenate source node, destination node, and edge embeddings
        sender_idx, receiver_idx = edge_matrix.T[:, 0], edge_matrix.T[:, 1]
        sender_feature = torch.index_select(input=node_feature, dim=0, index=sender_idx)
        receiver_feature = torch.index_select(
            input=node_feature, dim=0, index=receiver_idx
        )

        out = torch.cat([sender_feature, receiver_feature, edge_feature], -1)
        out = self.edge_mlp(out)
        # edge residual connection?
        # out += edge_feature

        return out


class NodeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        aggregation="sum",  # sum
        norm_type="LayerNorm",
    ):
        """
        Node processor: aggregate edge features to apply on nodes

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        """

        super(NodeProcessor, self).__init__()
        self.aggregation = aggregation
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(self, node_feature, edge_matrix, edge_feature):
        """node_feature: node feature: B x N_node x fdim
        edge:  [sender, receiver] edge_matrix, B x N_edge[2] x 2
        edge feature: B x N_edge[2] x fdim
        """
        receiver_idx = edge_matrix.T[:, 1]
        if self.aggregation == "sum":
            out = torch_scatter.scatter_add(edge_feature, receiver_idx, dim=0)
        elif self.aggregation == "max":
            out = torch_scatter.scatter_max(edge_feature, receiver_idx, dim=0)
        elif self.aggregation == "mean":
            out = torch_scatter.scatter_mean(edge_feature, receiver_idx, dim=0)
        else:
            raise Exception(f"Aggregation Operation {self.aggregation} is not Defined")

        out = torch.cat([node_feature, out], dim=-1)
        out = self.node_mlp(out)

        # add residual connection
        out += node_feature

        return out


class GraphNetBlock(nn.Module):
    """For one round of message passing,
    in_dim_node: input node feature dimension
    in_dim_edge: input edge feature dimension
    hidden_dim_node: number of nodes in a hidden layer for graph node processing
    hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
    hidden_layers_node: number of hidden layers for graph node processing
    hidden_layers_edge: number of hidden layers for graph edge processing
    """

    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type="LayerNorm",
    ):
        super().__init__()
        self.edge_model = EdgeProcessor(
            in_dim_node=in_dim_node,
            in_dim_edge=in_dim_edge,
            hidden_dim=hidden_dim_edge,
            hidden_layers=hidden_layers_edge,
            norm_type=norm_type,
        )
        self.node_model = NodeProcessor(
            in_dim_node=in_dim_node,
            in_dim_edge=in_dim_edge,
            hidden_dim=hidden_dim_node,
            hidden_layers=hidden_layers_node,
            norm_type=norm_type,
        )

    def forward(self, node_feature, edge_matrix, edge_feature):
        """node feature: [B x N_node x fdim]
        edge_matrix: [B x N_edge[2] x 2]
        edge_feature: [B x N_edge[2] x fdim]
        """

        # edge message passing
        edge_feature = self.edge_model(node_feature, edge_matrix, edge_feature)
        # node message passing
        node_feature = self.node_model(node_feature, edge_matrix, edge_feature)

        return node_feature, edge_feature


class Processor(nn.Module):
    def __init__(
        self,
        num_iterations=15,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type="LayerNorm",
    ):
        """
        Graph processor
        num_iterations: number of message-passing iterations (graph processor blocks)
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        """

        super(Processor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_iterations):
            self.blocks.append(
                GraphNetBlock(
                    in_dim_node=in_dim_node,
                    in_dim_edge=in_dim_edge,
                    hidden_dim_node=hidden_dim_node,
                    hidden_dim_edge=hidden_dim_edge,
                    hidden_layers_node=hidden_layers_node,
                    hidden_layers_edge=hidden_layers_edge,
                    norm_type=norm_type,
                )
            )

    def forward(self, x, edge_index, edge_feature):
        for block in self.blocks:
            x, edge_feature = block(x, edge_index, edge_feature)

        return x, edge_feature


# x_i^k = gammar(x_i^{k+1}, \sum \phi(x_i^{k-1}, x_j^{k-1}, (e_{j,i}))
# gammar: node_mlp; \phi: edge_mlp
class MeshMP(MessagePassing):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type=None,
    ):
        super(MeshMP, self).__init__(aggr="sum")
        # TODO: not sure if we need to use norm_type INSIDE the MLP, or outside, set None for now
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge,
            out_dim_edge,
            hidden_dim_edge,
            hidden_layers_edge,
            norm_type=None,
        )
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge,
            out_dim_node,
            hidden_dim_node,
            hidden_layers_node,
            norm_type=None,
        )

    def edge_update(self, edge_feature, x_j, x_i):
        # TODO add residual connection
        # return edge_feature + self.edge_mlp(torch.concat([edge_feature, x_j, x_i], dim=-1))
        return self.edge_mlp(torch.concat([edge_feature, x_j, x_i], dim=-1))

    def forward(self, x, edge_index, edge_feature):
        # TODO check if needed. If so, also find a way to add self-loops to edge_feature
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_feature = self.edge_updater(
            edge_index, edge_feature=edge_feature, x=x
        )  # call the edge_update()
        return self.propagate(edge_index, x=x, edge_feature=edge_feature), edge_feature

    def message(self, edge_feature):
        return edge_feature

    def update(self, aggr_out, x):
        # TODO add residual connection
        # return x + self.node_mlp(torch.cat([aggr_out, x], dim=-1))
        return self.node_mlp(torch.cat([aggr_out, x], dim=-1))


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, A, edge_index):
        # x has shape [N, feat_dim]
        # edge_index has shape [2, E]
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # the initial edge_index has already consider the self_loops
        # add MP with weights A
        x = self.propagate(edge_index, x=x, A=A)
        return x  # x is the aggregated message

    def message(self, x_j, A):
        # x_j has shape [E, feat_dim]
        # A: [E, 1] corresponding to the jth row, ith coloumn of original A matrix if define A^T x=y
        return A * x_j


class EdgeUpdater(MessagePassing):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type=None,
    ):
        super(EdgeUpdater, self).__init__(aggr="sum")

        self.node_mlp = MLP(
            in_dim_node, out_dim_node, hidden_dim_node, hidden_dim_node, norm_type=None
        )

        self.edge_mlp = MLP(
            in_dim_edge + 2 * hidden_dim_node,
            out_dim_edge,
            hidden_dim_edge,
            hidden_layers_edge,
            norm_type=None,
        )

    def forward(self, edge_index, edge_feature, x):
        # x has shape [N, feat_dim]
        # edge_index has shape [2, E]
        x = self.node_mlp(x)
        edge_feature = self.edge_updater(
            edge_index, edge_feature=edge_feature, x=x
        )  # call the edge_update()
        return edge_feature  # x is the aggregated message

    def edge_update(self, edge_feature, x_j, x_i):
        # TODO add residual connection
        # return edge_feature + self.edge_mlp(torch.concat([edge_feature, x_j, x_i], dim=-1))
        out = torch.concat([edge_feature, x_j, x_i], dim=-1)
        return self.edge_mlp(out)


class Net(nn.Module):
    def __init__(
        self,
        args,
        # data attributes:
        in_dim_node,
        in_dim_edge,
        out_dim,
        b_dim,
        num_edges=31260,
        # encoding attributes:
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        # graph processor attributes:
        num_iterations=30,
        hidden_dim_processor_node=128,
        hidden_dim_processor_edge=128,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        norm_type="LayerNorm",
        # decoder attributes:
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        dirichlet_idx=3,
        global_pool=False,
        # other:
        **kwargs,
    ):
        """
        MLP
        # input data:
        in_dim_node: mesh node attribute u_t
        in_dim_edge: mesh edge length
        out_dim: dim for u_t,
        b_dim: dim for x_t+1

        # encoder:
        out_dim_node out_dim_edge:
        hidden_dim_node, hidden_dim_edge: latent dimension
        hidden_layers_node, hidden_layers_edge: number of hidden layers

        # processor:
        num_iterations, number of message passing iterations
        hidden_dim_processor_node=128, hidden_dim_processor_edge: latent dimension
        hidden_layers_processor_node=2, hidden_layers_processor_edge: number of hidden layers

        # decoder:
        hidden_dim_decoder: latent dimension
        hidden_layers_decoder: number of hidden layers
        """

        super(Net, self).__init__()

        if "heat" in args.dataset:
            self.pde = "heat"
        elif "flow" in args.dataset:
            self.pde = "flow"
        elif "wave" in args.dataset:
            self.pde = "wave"
        else:
            self.pde = "syn"
        self.dirichlet_idx = dirichlet_idx
        self.node_encoder = MLP(
            in_dim_node, out_dim_node, hidden_dim_node, hidden_layers_node, norm_type
        )
        self.edge_encoder = MLP(
            in_dim_edge, out_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
        )
        # self.attention = nn.Linear(out_dim_node, 1)
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(
            MeshMP(
                out_dim_node,
                out_dim_edge,
                out_dim_node,
                out_dim_edge,
                hidden_dim_processor_node,
                hidden_dim_processor_edge,
                hidden_layers_processor_node,
                hidden_layers_processor_edge,
                norm_type,
            )
        )
        for i in range(num_iterations - 1):
            self.mp_layers.append(
                MeshMP(
                    out_dim_node,
                    out_dim_edge,
                    out_dim_node,
                    out_dim_edge,
                    hidden_dim_processor_node,
                    hidden_dim_processor_edge,
                    hidden_layers_processor_node,
                    hidden_layers_processor_edge,
                    norm_type,
                )
            )

        # decode graph.node_feature to b --> supervision
        # decode graph.edge_feature to A^-1, matrix multiplication and aggregation
        self.node_decoder_x = MLP(
            hidden_dim_processor_node,
            out_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            norm_type=None,
        )
        self.node_decoder_r = MLP(
            hidden_dim_processor_node,
            1,
            hidden_dim_decoder,
            hidden_layers_decoder,
            norm_type=None,
        )
        edge_decoder_dim = hidden_dim_processor_edge
        if args.use_global:
            self.avgp = torch.nn.AvgPool1d(num_edges, num_edges)
            edge_decoder_dim += hidden_dim_processor_edge
        self.edge_decoder_L = MLP(
            edge_decoder_dim,
            1,
            hidden_dim_decoder,
            hidden_layers_decoder,
            norm_type=None,
        )  # must be None
        self.edge_decoder_D = MLP(
            edge_decoder_dim,
            1,
            hidden_dim_decoder,
            hidden_layers_decoder,
            norm_type=None,
        )  # must be None
        self.postprocess = MP()  # non parameter

    def forward(
        self,
        node_attr,
        edge_attr,
        edge_index,
        diag=None,
        input_r=None,
        input_x=None,
        batch_idx=None,
        include_r=False,
        use_global=False,
        diagonalize=False,
        use_pred_x=False,
    ):
        node_encoder_feature = self.node_encoder(node_attr)
        edge_encoder_feature = self.edge_encoder(edge_attr)
        dirichlet_mask = node_attr[:, self.dirichlet_idx]  # [bs*num_nodes, ]
        dirichlet_mask = dirichlet_mask.to(torch.bool)
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr=edge_attr)
        zero_mask = torch.logical_or(
            dirichlet_mask[edge_index[0, :]], dirichlet_mask[edge_index[1, :]]
        )

        x = node_encoder_feature
        edge_feature = edge_encoder_feature

        for mp_l in self.mp_layers:
            x, edge_feature = mp_l(x, edge_index, edge_feature)
        decoded_x = self.node_decoder_x(x)

        if include_r:
            decoded_r = self.node_decoder_r(x)
            r_x = input_x * decoded_r
            r_x[dirichlet_mask] = 0.0
        else:
            decoded_r = torch.zeros_like(input_r)
            r_x = 0.0

        if use_global:
            edge_avg = self.avgp(edge_feature.permute(1, 0))  # 16, batchsize
            batch_size = len(torch.unique(batch_idx))
            num_edges = edge_feature.shape[0] // batch_size
            edge_avg_pad = torch.repeat_interleave(
                edge_avg,
                torch.tensor([num_edges] * batch_size, device=edge_attr.device),
                dim=-1,
            )
            global_padded_edge_feature = torch.cat(
                [edge_feature, edge_avg_pad.permute(1, 0)], dim=-1
            )
            decoded_L = self.edge_decoder_L(global_padded_edge_feature)
        else:
            decoded_L = self.edge_decoder_L(edge_feature)  # [E, 1]

        factor = 1.0
        if self.pde == "heat":
            diag_ele = edge_attr[:, -1] + edge_attr[:, -2]
            diag_ele = diag_ele[(edge_index[0, :] == edge_index[1, :]).T]
        elif self.pde == "flow":
            diag_ele = edge_attr[:, -1]
            diag_ele = diag_ele[(edge_index[0, :] == edge_index[1, :]).T]
        elif self.pde == "wave":
            diag_ele = edge_attr[:, 1]
            diag_ele = diag_ele[(edge_index[0, :] == edge_index[1, :]).T]
            factor = 100
            # diag_ele[dirichlet_node]=1.0
        else:
            diag_ele = edge_attr[:, -1]
            diag_ele = diag_ele[(edge_index[0, :] == edge_index[1, :])]
        diag_ele = diag_ele.reshape(-1, 1)

        mean_edge_index, decoded_L_mean = torch_geometric.utils.to_undirected(
            edge_index, decoded_L, reduce="mean"
        )
        # to enforce a diagonal validity of the matrix
        decoded_L_mean[mean_edge_index.T[:, 0] < mean_edge_index.T[:, 1]] = 0
        decoded_L_mean[mean_edge_index.T[:, 0] == (mean_edge_index.T[:, 1])] = (
            torch.sqrt(diag_ele)
        )
        decoded_edge_indices = mean_edge_index
        decoded_L = decoded_L_mean * factor

        if not use_pred_x:
            decoded_x = input_x
        LTx = self.postprocess(decoded_x, decoded_L, decoded_edge_indices)

        swap_mapping = torch.tensor([[0, 1], [1, 0]]).to(edge_index.device)
        trans_edge_index = decoded_edge_indices.clone()
        trans_edge_index[swap_mapping[:, 0]] = decoded_edge_indices[swap_mapping[:, 1]]
        LLTx = self.postprocess(LTx, decoded_L, trans_edge_index)

        b_pred_flattened = LLTx / factor  # + Dx * decoded_x
        if self.pde == "wave":
            b_pred_flattened[dirichlet_mask] = 0
        else:
            b_pred_flattened[dirichlet_mask] = input_x[dirichlet_mask]

        if not use_pred_x:
            output_x = torch.zeros_like(input_x)
        else:
            output_x = decoded_x

        reverse_factor = math.sqrt(1 / factor)
        return (
            b_pred_flattened,
            ((decoded_L, diag_ele, reverse_factor), decoded_edge_indices),
            output_x,
        )
