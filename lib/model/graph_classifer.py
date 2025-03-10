import torch
import torch.nn as nn

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, lstm_hidden_dim, output_dim):
        """
        input_dim: Düğüm özelliklerinin boyutu
        hidden_dim_1: İlk konvolüsyon katmanının çıkış boyutu (GATConv)
        hidden_dim_2: İkinci konvolüsyon katmanının çıkış boyutu (GCNConv)
        lstm_hidden_dim: LSTM katmanının hidden boyutu
        output_dim: Sınıflandırma çıktısı (örneğin binary için 2)
        """
        super(GraphClassifier, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)
        # LSTM: batch_first=True, tek katman, unidirectional
        self.lstm = nn.LSTM(input_size=hidden_dim_2, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph convolution işlemleri
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # to_dense_batch ile her graph'ın düğümlerini dense formata dönüştürüyoruz.
        # padded_x: [B, T, hidden_dim_2] şeklinde, B: batch size, T: en fazla düğüm sayısı
        # mask: [B, T] gerçek düğüm konumlarını işaretler.
        padded_x, mask = to_dense_batch(x, batch)
        # Her graph için gerçek düğüm sayısını hesaplıyoruz.
        lengths = mask.sum(dim=1).long()

        # LSTM'e vermek için sequence'i pack ediyoruz.
        # enforce_sorted=False ile sıralama zorunluluğunu kaldırıyoruz.
        packed_input = pack_padded_sequence(padded_x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # h_n: [num_layers * num_directions, B, lstm_hidden_dim]. Tek katmanlı unidirectional LSTM'de
        # h_n[-1] her graph için son hidden state'i verir.
        graph_rep = h_n[-1]

        # Fully-connected katman ile sınıflandırma
        out = self.fc(graph_rep)
        return out
