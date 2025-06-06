import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, GlobalAttention
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence

class GoBERT(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout=0.5):
        """
        input_dim: Düğüm özelliklerinin boyutu (FinBERT embedding boyutu)
        hidden_dim_1: İlk GATConv katmanının çıkış boyutu (her head için)
        hidden_dim_2: İkinci GCNConv katmanının çıkış boyutu
        output_dim: Sınıflandırma çıktısı (örneğin, merge: 1, non-merge: 0 için 2)
        dropout: Dropout oranı
        """
        super(GoBERT, self).__init__()
        # İlk katmanda 4 head kullanarak multi-head GAT ile daha zengin özellik çıkarımı sağlıyoruz.
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1, heads=4, concat=True)
        # Batch normalization; çıkış boyutu = hidden_dim_1 * 4
        self.bn1 = nn.BatchNorm1d(hidden_dim_1 * 4)

        # İkinci katmanda GCNConv
        self.conv2 = GCNConv(hidden_dim_1 * 4, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)

        # Global attention pooling: Düğüm özelliklerinin ağırlıklı olarak birleştirilmesi
        self.att_pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim_2, 1)))

        # Final fully-connected katman
        self.fc = nn.Linear(hidden_dim_2, output_dim)

        # Dropout katmanı
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # İlk konvolüsyon: GAT ile özellik çıkarımı, ELU aktivasyon, BatchNorm ve dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        # İkinci konvolüsyon: GCN, ELU aktivasyon, BatchNorm ve dropout
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        # Global attention pooling ile graf temsilini elde etme
        graph_rep = self.att_pool(x, batch)

        # Final sınıflandırma katmanı
        out = self.fc(graph_rep)
        out = out.squeeze(1)
        return out
