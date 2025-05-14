import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse


class Hyper_GAT_Model(nn.Module):
    def __init__(self, input_dim, gat_hidden_dim=80, output_dim=1, gat_heads=8, gat_layers=1):
        super(Hyper_GAT_Model, self).__init__()
        self.num_electrodes = 19
        self.gat_heads = gat_heads
        self.input_dim = input_dim 
        self.gat_layers = gat_layers
        self.gats = nn.ModuleList([
            GATv2Conv(
                in_channels=input_dim if i == 0 else gat_hidden_dim * gat_heads,
                out_channels=gat_hidden_dim,
                heads=gat_heads,
                concat=True
            ) for i in range(gat_layers)
        ])
        self.attention_pool = nn.Linear(gat_hidden_dim * gat_heads, 1)
        self.fc = nn.Linear(gat_hidden_dim * gat_heads, output_dim)

    def build_spatiotemporal_graph(self, num_nodes, num_windows):
        total_nodes = num_nodes * num_windows
        adj = torch.zeros((total_nodes, total_nodes))

        # Step 1: Intra-window (spatial) connections
        for w in range(num_windows):
            start = w * num_nodes
            end = start + num_nodes
            adj[start:end, start:end] = 1  # fully connect within window

        # Step 2: Inter-window (temporal) connections
        for w in range(num_windows - 1):
            for n in range(num_nodes):
                src = w * num_nodes + n
                dst = (w + 1) * num_nodes + n
                adj[src, dst] = 1
                adj[dst, src] = 1  # make it bidirectional if needed


        edge_index, _ = dense_to_sparse(adj)
        return edge_index  # shape (2, num_edges)


    def forward(self, x, return_debug=False):
        """
        Args:
            x: Tensor of shape (batch_size, num_freqs, num_electrodes)
            return_debug: If True, also returns intermediate tensors for debugging
        Returns:
            logits: (batch_size,) or (logits, debug_dict)
        """
        batch_size, num_windows, num_freqs, num_electrodes = x.shape
        edge_index = self.build_spatiotemporal_graph(num_electrodes, num_windows).to(x.device)

        graph_outputs = []

        for b in range(batch_size):
            nodes = x[b].permute(0,2, 1)  # shape: (num_windows, num_electrodes, num_freqs)
            nodes = nodes.reshape(num_windows * num_electrodes, num_freqs)  # shape: (num_windows * num_electrodes, num_freqs)
            out = nodes
            for i in range(self.gat_layers):
                out = self.gats[i](out, edge_index)
                out = F.elu(out)
            attn_scores = self.attention_pool(out)  # (num_nodes, 1)
            attn_scores = torch.softmax(attn_scores, dim=0)  # (num_nodes, 1)
            pooled = (attn_scores * out).sum(dim=0)  # weighted sum
            graph_outputs.append(pooled)

        graph_outputs = torch.stack(graph_outputs, dim=0)  # (batch_size, gat_hidden_dim * gat_heads)
        logits = self.fc(graph_outputs)  # (batch_size, 1)

        return logits.squeeze(1)  # (batch_size,)


