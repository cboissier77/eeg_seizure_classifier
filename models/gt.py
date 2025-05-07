import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP classification layer
class MLP_layer(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super(MLP_layer, self).__init__()
        layers = [nn.Linear(input_dim, input_dim) for _ in range(L)]
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.L = L

    def forward(self, x):
        for l in range(self.L):
            x = F.relu(self.layers[l](x))
        return self.layers[self.L](x)

# Multi-head attention layer (manual)
class graph_MHA_layer(nn.Module):
    def __init__(self, hidden_dim, head_hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_hidden_dim = head_hidden_dim
        self.WQ = nn.Linear(hidden_dim, head_hidden_dim * num_heads)
        self.WK = nn.Linear(hidden_dim, head_hidden_dim * num_heads)
        self.WV = nn.Linear(hidden_dim, head_hidden_dim * num_heads)

    def forward(self, edge_index, h):
        Q = self.WQ(h).view(-1, self.num_heads, self.head_hidden_dim)
        K = self.WK(h).view(-1, self.num_heads, self.head_hidden_dim)
        V = self.WV(h).view(-1, self.num_heads, self.head_hidden_dim)

        src, dst = edge_index
        q_dst = Q[dst]
        k_src = K[src]
        v_src = V[src]

        attn_score = (q_dst * k_src).sum(dim=-1) / (self.head_hidden_dim ** 0.5)
        attn_score = attn_score.clamp(-5, 5).softmax(dim=0)

        h_message = attn_score.unsqueeze(-1) * v_src

        h_out = torch.zeros_like(Q)
        h_out.index_add_(0, dst, h_message)

        return h_out.view(-1, self.num_heads * self.head_hidden_dim)

# Graph Transformer Block
class GraphTransformer_layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, norm='LN', dropout=0.0):
        super().__init__()
        self.gMHA = graph_MHA_layer(hidden_dim, hidden_dim // num_heads, num_heads)
        self.WO = nn.Linear(hidden_dim, hidden_dim)

        if norm == 'LN':
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        else:
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, edge_index, h):
        h_rc = h
        h = self.gMHA(edge_index, h)
        h = self.WO(F.dropout(h, self.dropout, training=self.training))
        h = h_rc + h
        h = self.norm1(h)

        h_rc = h
        h = F.relu(self.linear1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear2(h)
        h = h_rc + h
        h = self.norm2(h)
        return h

# Full Graph Transformer Network
class EEG_GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, norm='LN', L=3, pos_enc_dim=8):
        super().__init__()
        self.num_nodes = 19
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pos_enc_dim = pos_enc_dim

        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_pe = nn.Linear(pos_enc_dim, hidden_dim)

        self.GraphTransformer_layers = nn.ModuleList([
            GraphTransformer_layer(hidden_dim, num_heads, norm) for _ in range(L)
        ])

        self.MLP_layer = MLP_layer(hidden_dim, output_dim)

        self.edge_index = self.build_fully_connected_graph()
        self.pe = self.compute_positional_encoding()

    def build_fully_connected_graph(self):
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def compute_positional_encoding(self):
        A = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    A[i, j] = 1
        D = torch.diag(A.sum(dim=1))
        L = D - A

        eigvals, eigvecs = torch.linalg.eigh(L)
        return eigvecs[:, 1:self.pos_enc_dim+1]

    def forward(self, node_features):
        """
        node_features: (batch_size, input_dim, num_nodes)
        """
        batch_size, input_dim, num_nodes = node_features.shape
        assert num_nodes == self.num_nodes, f"Expected {self.num_nodes} electrodes, got {num_nodes}"

        node_features = node_features.permute(0, 2, 1)  # now (batch_size, num_nodes, input_dim)

        edge_index = self.edge_index.to(node_features.device)
        pe = self.pe.to(node_features.device)

        graph_outputs = []
        for b in range(batch_size):
            h_feat = self.embedding_h(node_features[b])  # (19, hidden_dim)
            h_pe = self.embedding_pe(pe)  # (19, hidden_dim)
            h = h_feat + h_pe

            for layer in self.GraphTransformer_layers:
                h = layer(edge_index, h)

            pooled = h.mean(dim=0)  # (hidden_dim,)
            graph_outputs.append(pooled)

        graph_outputs = torch.stack(graph_outputs, dim=0)  # (batch_size, hidden_dim)
        output = self.MLP_layer(graph_outputs)  # (batch_size, output_dim)
        return output
