import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

class SentenceClassifier(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, fc1_dim, fc2_dim, output_dim):
        """
        input_dim: Her cümlenin embedding boyutu.
        lstm_hidden_dim: LSTM katmanının gizli boyutu.
        fc1_dim: Birinci fully connected katmanın çıkış boyutu.
        fc2_dim: İkinci fully connected katmanın çıkış boyutu.
        output_dim: Son sınıflandırma çıktısı (örneğin, binary için 2).
        """
        super(SentenceClassifier, self).__init__()
        # LSTM: batch_first=True, tek katman, unidirectional
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        # Üç adet fully connected katman
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)

    def forward(self, embeddings, lengths):
        """
        embeddings: [batch_size, seq_len, input_dim] boyutunda padded cümle embedding tensörü.
        lengths: [batch_size] boyutunda her örnekteki gerçek cümle sayısını belirten tensor.
        """
        # Pack edilmiş sequence ile LSTM'e veriyoruz.
        packed_input = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # Tek katmanlı unidirectional LSTM'de h_n[-1] her örneğin son hidden state'idir.
        lstm_out = h_n[-1]  # [batch_size, lstm_hidden_dim]

        # Fully-connected katmanlar üzerinden geçirme
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x