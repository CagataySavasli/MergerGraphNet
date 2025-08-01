import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class Q_RoBERT(nn.Module):
    def __init__(self, input_dim=768, quant_input_dim=63, hidden_dim=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # BERT sequence LSTM
        self.text_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Quantitative data LSTM
        self.quant_lstm = nn.LSTM(
            input_size=quant_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        direction = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction * 2, 1)  # 2 for two LSTMs

    def forward(self, sequences, quantative_data):
        # sequences: list of [seq_len_i, input_dim]
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n_text, _) = self.text_lstm(packed)

        if self.bidirectional:
            h_n_text = h_n_text.view(self.num_layers, 2, -1, self.hidden_dim)
            last_text = h_n_text[-1]
            hidden_text = torch.cat([last_text[0], last_text[1]], dim=1)
        else:
            hidden_text = h_n_text[-1]

        # quantative_data: Tensor of shape [batch_size, 12, 63]
        _, (h_n_quant, _) = self.quant_lstm(quantative_data)

        if self.bidirectional:
            h_n_quant = h_n_quant.view(self.num_layers, 2, -1, self.hidden_dim)
            last_quant = h_n_quant[-1]
            hidden_quant = torch.cat([last_quant[0], last_quant[1]], dim=1)
        else:
            hidden_quant = h_n_quant[-1]

        # Combine both hidden states
        combined = torch.cat([hidden_text, hidden_quant], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
