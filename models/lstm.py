import torch
import torch.nn as nn

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


class EEG_LSTM_Model(nn.Module):
    def __init__(self, input_dim=1, lstm_hidden_dim=32, output_dim=1,lstm_layers=1):
        super(EEG_LSTM_Model, self).__init__()
        self.num_electrodes = 19
        self.bidirectional = True
        self.lstm_hidden_dim = lstm_hidden_dim * (2 if self.bidirectional else 1)

        self.lstm_modules = nn.ModuleList([
            ElectrodeLSTM(input_dim, lstm_hidden_dim, bidirectional=self.bidirectional, lstm_layers=lstm_layers)
            for _ in range(self.num_electrodes)
        ])

        self.fc = nn.Linear(self.lstm_hidden_dim, output_dim)
  

    def forward(self, x):
        # x: (batch, seq_len=354, num_electrodes=19)
        batch_size, seq_len, num_electrodes = x.shape
        assert num_electrodes == self.num_electrodes, "Input should have 19 electrodes"
        lstm_outputs = []
        for i in range(self.num_electrodes):
            electrode_input = x[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)
            out = self.lstm_modules[i](electrode_input)  # (batch, hidden_dim * 2)
            lstm_outputs.append(out)

        # average the outputs from all electrodes 
        lstm_outputs = torch.stack(lstm_outputs, dim=1)  # (batch, num_electrodes, hidden_dim * 2)
        lstm_outputs = lstm_outputs.mean(dim=1)

        # pass in fully connected layer
        logits = self.fc(lstm_outputs)

        return logits.squeeze(1)  # (batch,)
