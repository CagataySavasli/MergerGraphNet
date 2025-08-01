import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class Q_ToBERT(nn.Module):
    def __init__(
        self,
        input_dim=768,
        quant_input_dim=63,
        transformer_heads=8,
        transformer_layers=2,
        dim_feedforward=2048,
        lstm_hidden_dim=128,
        lstm_layers=1,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # LSTM for quantitative input
        self.quant_lstm = nn.LSTM(
            input_size=quant_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Combined classifier
        self.classifier = nn.Linear(input_dim + lstm_hidden_dim, 1)

    def forward(self, sequences, quantative_data):
        device = sequences[0].device
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long, device=device)
        padded = pad_sequence(sequences, batch_first=True).to(device)

        mask = torch.arange(padded.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = padded * math.sqrt(self.input_dim)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=mask)

        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        sums = out.sum(dim=1)
        counts = (~mask).sum(dim=1).unsqueeze(1)
        text_hidden = sums / counts

        _, (h_n_quant, _) = self.quant_lstm(quantative_data)
        quant_hidden = h_n_quant[-1]

        combined = torch.cat([text_hidden, quant_hidden], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
