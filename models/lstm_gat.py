import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class ElectrodeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=1, bidirectional=True):
        super(ElectrodeLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # (batch, hidden_dim * 2)
        else:
            out = hn[-1]
        return self.norm(out)


class EEG_LSTM_GAT_Model(nn.Module):
    def __init__(self, input_dim=1, lstm_hidden_dim=32, gat_hidden_dim=64, output_dim=1, gat_heads=8,lstm_layers=1, fully_connected=True):
        super(EEG_LSTM_GAT_Model, self).__init__()
        self.num_electrodes = 19
        self.bidirectional = True
        self.lstm_hidden_dim = lstm_hidden_dim * (2 if self.bidirectional else 1)

        self.lstm_modules = nn.ModuleList([
            ElectrodeLSTM(input_dim, lstm_hidden_dim, bidirectional=self.bidirectional, lstm_layers=lstm_layers)
            for _ in range(self.num_electrodes)
        ])

        self.gat_heads = gat_heads
        self.gat = GATv2Conv(
            in_channels=self.lstm_hidden_dim,
            out_channels=gat_hidden_dim,
            heads=gat_heads,
            concat=True  # default: output shape is [N, heads * out_channels]
        )

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

        # Convert label edges to index edges
        label_to_index = {label: idx for idx, label in enumerate(electrode_labels)}
        edge_index = []

        for src_label, dst_label in edges:
            src = label_to_index[src_label]
            dst = label_to_index[dst_label]
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # add reverse for undirected graph

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Shape [2, num_edges]
        return edge_index


    def forward(self, x):
        # x: (batch, seq_len=354, num_electrodes=19)
        batch_size, seq_len, num_electrodes = x.shape
        assert num_electrodes == self.num_electrodes, "Input should have 19 electrodes"

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        lstm_outputs = []
        for i in range(self.num_electrodes):
            electrode_input = x[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)
            out = self.lstm_modules[i](electrode_input)  # (batch, hidden_dim * 2)
            lstm_outputs.append(out)

        node_features = torch.stack(lstm_outputs, dim=1)  # (batch, 19, hidden_dim * 2)

        graph_outputs = []
        edge_index = self.edge_index.to(node_features.device)
        for b in range(batch_size):
            out = self.gat(node_features[b], edge_index)  # (19, gat_hidden_dim * heads)
            pooled = out.mean(dim=0) #self.pool(out)   #out.mean(dim=0)  # global mean pooling
            graph_outputs.append(pooled)

        graph_outputs = torch.stack(graph_outputs, dim=0)  # (batch, gat_hidden_dim * heads)
        logits = self.fc(graph_outputs)  # (batch, 1)

        return logits.squeeze(1)  # (batch,)

    def load_and_freeze_lstm(self, path_to_pth):
        state_dict = torch.load(path_to_pth, map_location='cpu')
        
        for i in range(self.num_electrodes):
            # Extract keys for lstm_modules.i.*
            submodule_prefix = f"lstm_modules.{i}."
            lstm_state_dict = {
                key[len(submodule_prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(submodule_prefix)
            }

            # Load weights into corresponding LSTM module
            self.lstm_modules[i].load_state_dict(lstm_state_dict)

            # Freeze the weights
            for param in self.lstm_modules[i].parameters():
                param.requires_grad = False

