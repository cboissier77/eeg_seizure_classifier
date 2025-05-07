import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class EEG_GAT_Model(nn.Module):
    def __init__(self, input_dim=1, gat_hidden_dim=64, output_dim=1, gat_heads=8, fully_connected=True):
        super(EEG_GAT_Model, self).__init__()
        self.num_electrodes = 19
        self.gat_heads = gat_heads
        self.input_dim = input_dim

        # Two GAT layers
        self.gat1 = GATv2Conv(in_channels=input_dim, out_channels=gat_hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATv2Conv(in_channels=gat_hidden_dim * gat_heads, out_channels=gat_hidden_dim, heads=gat_heads, concat=True)

        # Attention pooling layer
        self.attention_pool = nn.Linear(gat_hidden_dim * gat_heads, 1)

        self.fc = nn.Linear(gat_hidden_dim * gat_heads, output_dim)

        self.edge_index = self.build_fully_connected_graph(self.num_electrodes) if fully_connected else self.build_eeg_graph()

    def build_fully_connected_graph(self, num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def build_eeg_graph(self):
        electrode_labels = [
            "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3",
            "P4", "O1", "O2", "F7", "F8",
            "T3", "T4", "T5", "T6", "Fz",
            "Cz", "Pz"
        ]

        edges = [
            ("Fp1", "F7"), ("Fp1", "F3"), ("Fp1", "Fz"),
            ("Fp2", "F4"), ("Fp2", "F8"), ("Fp2", "Fz"),
            ("F7", "F3"), ("F3", "Fz"), ("Fz", "F4"), ("F4", "F8"),
            ("F7", "T3"), ("F3", "C3"), ("Fz", "Cz"), ("F4", "C4"), ("F8", "T4"),
            ("T3", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T4"),
            ("T3", "T5"), ("C3", "P3"), ("Cz", "Pz"), ("C4", "P4"), ("T4", "T6"),
            ("T5", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "T6"),
            ("T5", "O1"), ("P3", "O1"), ("P4", "O2"), ("T6", "O2")
        ]

        label_to_index = {label: idx for idx, label in enumerate(electrode_labels)}
        edge_index = []

        for src_label, dst_label in edges:
            src = label_to_index[src_label]
            dst = label_to_index[dst_label]
            edge_index.append([src, dst])
            edge_index.append([dst, src])

        return torch.tensor(edge_index, dtype=torch.long).t()

    def forward(self, x):
        # x: (batch_size, seq_len, num_electrodes)
        batch_size, seq_len, num_electrodes = x.shape
        assert num_electrodes == self.num_electrodes, f"Expected {self.num_electrodes} electrodes, got {num_electrodes}"

        # (batch_size, num_electrodes, seq_len)
        node_features = x.permute(0, 2, 1)

        graph_outputs = []
        edge_index = self.edge_index.to(node_features.device)
        for b in range(batch_size):
            out = self.gat1(node_features[b], edge_index)
            out = F.elu(out)
            out = self.gat2(out, edge_index)

            # Attention pooling
            attn_scores = self.attention_pool(out)  # (num_nodes, 1)
            attn_scores = torch.softmax(attn_scores, dim=0)  # (num_nodes, 1)
            pooled = (attn_scores * out).sum(dim=0)  # weighted sum

            graph_outputs.append(pooled)

        graph_outputs = torch.stack(graph_outputs, dim=0)  # (batch_size, gat_hidden_dim * gat_heads)
        logits = self.fc(graph_outputs)  # (batch_size, 1)

        return logits.squeeze(1)  # (batch_size,)
