import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, input_dim, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, L, C = x.shape
        assert L % self.patch_size == 0, "Sequence length must be divisible by patch size"
        x = x.view(B, L // self.patch_size, self.patch_size * C)  # (batch, num_patches, patch_size * input_dim)
        x = self.proj(x)  # (batch, num_patches, embed_dim)
        return x


class ElectrodeTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, num_layers=2, nhead=4):
        super(ElectrodeTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(patch_size, input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        x = self.transformer_encoder(x)  # (batch, num_patches, embed_dim)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.norm(x)


class EEG_Transformer_Model(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, output_dim=1, patch_size=10, num_layers=2, nhead=4):
        super(EEG_Transformer_Model, self).__init__()
        self.num_electrodes = 19

        self.transformer_modules = nn.ModuleList([
            ElectrodeTransformer(input_dim, embed_dim, patch_size, num_layers=num_layers, nhead=nhead)
            for _ in range(self.num_electrodes)
        ])

        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, num_electrodes)
        batch_size, seq_len, num_electrodes = x.shape
        assert num_electrodes == self.num_electrodes, "Input should have 19 electrodes"

        transformer_outputs = []
        for i in range(self.num_electrodes):
            electrode_input = x[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)
            out = self.transformer_modules[i](electrode_input)  # (batch, embed_dim)
            transformer_outputs.append(out)

        # average the outputs from all electrodes 
        transformer_outputs = torch.stack(transformer_outputs, dim=1)  # (batch, num_electrodes, embed_dim)
        transformer_outputs = transformer_outputs.mean(dim=1)

        # pass in fully connected layer
        logits = self.fc(transformer_outputs)

        return logits.squeeze(1)  # (batch,)
