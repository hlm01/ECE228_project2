from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin_edge = Linear(in_channels + edge_dim, out_channels, bias=False)
        self.lin_out = Linear(out_channels, out_channels, bias=False)

        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin_edge.weight)
        glorot(self.lin_out.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        return out + self.bias

    def edge_update(
        self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
        index: Tensor, ptr: OptTensor, size_i: Optional[int]
    ) -> Tensor:
        edge_input = torch.cat([x_j, edge_attr], dim=-1)
        edge_msg = F.leaky_relu(self.lin_edge(edge_input), negative_slope=0.01)

        alpha_j = (edge_msg @ self.att_l.T).squeeze(-1)
        alpha_i = (x_i @ self.att_r.T).squeeze(-1)
        alpha = F.leaky_relu(alpha_j + alpha_i)

        alpha = softmax(alpha, index, ptr, size_i)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin_out(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.input_proj = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gate_gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False)
            for _ in range(num_layers - 1)
        ])
        self.atom_grus = torch.nn.ModuleList([
            GRUCell(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)
        ])

        self.mol_conv = GATConv(hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.readout_proj = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gate_gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.readout_proj.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        # Step 1: Atom-level encoding
        x = F.leaky_relu(self.input_proj(x), negative_slope=0.01)
        h = F.elu(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gate_gru(h, x)
        x = F.relu(x)

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x)
            x = F.relu(x)

        # Step 2: Molecule-level iterative readout
        mol_embed = global_add_pool(x, batch)
        mol_embed = F.relu(mol_embed)

        virtual_edge_index = torch.stack([torch.arange(batch.size(0), device=batch.device), batch], dim=0)

        for _ in range(self.num_timesteps):
            h = self.mol_conv((x, mol_embed), virtual_edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            mol_embed = self.mol_gru(h, mol_embed)
            mol_embed = F.relu(mol_embed)

        # Final prediction
        mol_embed = F.dropout(mol_embed, p=self.dropout, training=self.training)
        return self.readout_proj(mol_embed)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"edge_dim={self.edge_dim}, "
                f"num_layers={self.num_layers}, "
                f"num_timesteps={self.num_timesteps})")