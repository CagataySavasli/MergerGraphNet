import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class SentenceEmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, lstm_hidden_dim, output_dim):
        """
        input_dim: Cümle embedding'lerinin boyutu.
        fc1_dim: FC1 katmanının çıkış boyutu.
        fc2_dim: FC2 katmanının çıkış boyutu.
        lstm_hidden_dim: LSTM katmanının hidden boyutu.
        output_dim: Sınıflandırma çıktısı (örneğin, sınıf sayısı).
        """
        super(SentenceEmbeddingClassifier, self).__init__()
        # LSTM, batch_first=True olarak tanımlandı.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)

    def forward(self, padded_x, lengths):
        packed_input = pack_padded_sequence(padded_x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out = h_n[-1]
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        out = self.fc3(x)
        return out
