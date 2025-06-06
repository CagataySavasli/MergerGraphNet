import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# --- Model Definition ---
class RoBERT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        direction = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction, 1)

    def forward(self, sequences):
        # sequences: list of tensors [seq_len_i, input_dim]
        # lengths must be CPU int64 tensor for pack_padded_sequence
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        # get last hidden state
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)
            last = h_n[-1]
            hidden = torch.cat([last[0], last[1]], dim=1)
        else:
            hidden = h_n[-1]

        logits = self.classifier(hidden).squeeze(1)
        return logits