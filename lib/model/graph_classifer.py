import torch
import torch.nn as nn

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, GlobalAttention
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        """
        input_dim: Düğüm özelliklerinin boyutu
        hidden_dim_1: İlk konvolüsyon katmanının çıkış boyutu (GATConv)
        hidden_dim_2: İkinci konvolüsyon katmanının çıkış boyutu (GCNConv)
        output_dim: Sınıflandırma çıktısı (örneğin binary için 2)
        """
        super(GraphClassifier, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)
        # Global pooling sonrası çıkış boyutu hidden_dim_2 olur
        self.fc = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Graph convolution işlemleri
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Her grafın düğüm özelliklerini global mean pooling ile birleştiriyoruz.
        graph_rep = global_mean_pool(x, batch)

        # Fully-connected katman ile sınıflandırma
        out = self.fc(graph_rep)
        return out


class MultiheadGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout=0.5):
        """
        input_dim: Düğüm özelliklerinin boyutu (FinBERT embedding boyutu)
        hidden_dim_1: İlk GATConv katmanının çıkış boyutu (her head için)
        hidden_dim_2: İkinci GCNConv katmanının çıkış boyutu
        output_dim: Sınıflandırma çıktısı (örneğin, merge: 1, non-merge: 0 için 2)
        dropout: Dropout oranı
        """
        super(MultiheadGraphClassifier, self).__init__()
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
        return out

class GraphResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):
        super(GraphResidualClassifier, self).__init__()

        # İlk katmanda GAT ve GCN birlikte kullanılacak
        self.gat1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1)
        self.gcn1 = GCNConv(input_dim, hidden_dim_1)
        self.lin1 = nn.Linear(input_dim, hidden_dim_1)  # Residual için lineer dönüşüm

        # İkinci katmanda yine hem GAT hem GCN
        self.gat2 = GATConv(in_channels=hidden_dim_1, out_channels=hidden_dim_2)
        self.gcn2 = GCNConv(hidden_dim_1, hidden_dim_2)
        self.lin2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # Son katmanda sınıflandırma için GAT ve GCN birlikte kullanılacak
        self.gat3 = GATConv(in_channels=hidden_dim_2, out_channels=hidden_dim_3)
        self.gcn3 = GCNConv(hidden_dim_2, hidden_dim_3)
        self.lin3 = nn.Linear(hidden_dim_2, hidden_dim_3)

        self.lin4 = nn.Linear(hidden_dim_3, output_dim)


    def forward(self, data):
        # İlk katman
        x, edge_index, batch = data.x, data.edge_index, data.batch

        gat_out1 = self.gat1(x, edge_index)
        gcn_out1 = self.gcn1(x, edge_index)
        x = torch.relu(gat_out1 + gcn_out1 + self.lin1(x))  # Residual connection
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # İkinci katman
        gat_out2 = self.gat2(x, edge_index)
        gcn_out2 = self.gcn2(x, edge_index)
        x = torch.relu(gat_out2 + gcn_out2 + self.lin2(x))  # Residual connection
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # Son katman (sınıflandırma için)
        gat_out3 = self.gat3(x, edge_index)
        gcn_out3 = self.gcn3(x, edge_index)
        x = gat_out3 + gcn_out3 + self.lin3(x)  # Residual connection

        x = global_mean_pool(x, batch)

        out = self.lin4(x)

        return out

class GraphLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, lstm_hidden_dim, output_dim):
        """
        input_dim: Düğüm özelliklerinin boyutu
        hidden_dim_1: İlk konvolüsyon katmanının çıkış boyutu (GATConv)
        hidden_dim_2: İkinci konvolüsyon katmanının çıkış boyutu (GCNConv)
        lstm_hidden_dim: LSTM katmanının hidden boyutu
        output_dim: Sınıflandırma çıktısı (örneğin binary için 2)
        """
        super(GraphLSTMClassifier, self).__init__()
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

class GraphTransClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, nhead, num_encoder_layers, output_dim):
        """
        input_dim: Düğüm özelliklerinin boyutu
        hidden_dim_1: İlk konvolüsyon katmanının çıkış boyutu (GATConv)
        hidden_dim_2: İkinci konvolüsyon katmanının çıkış boyutu (GCNConv) ve Transformer'ın d_model değeri
        nhead: Transformer Encoder katmanında kullanılacak baş sayısı
        num_encoder_layers: Transformer Encoder katman sayısı
        output_dim: Sınıflandırma çıktısı (örneğin binary için 2)
        """
        super(GraphTransClassifier, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)

        # Transformer Encoder: d_model olarak hidden_dim_2 kullanılıyor.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim_2, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graf konvolüsyon işlemleri
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # to_dense_batch ile her grafın düğümleri dense formata dönüştürülüyor.
        # padded_x: [B, T, hidden_dim_2] şeklinde, mask: [B, T] (True: gerçek düğüm)
        padded_x, mask = to_dense_batch(x, batch)

        # Transformer, padding yapılan kısımları görmezden gelmesi için src_key_padding_mask istiyor.
        # mask: True olan yerler gerçek veriyi ifade eder; Transformer'a True, pad olan yerler için verilir.
        src_key_padding_mask = ~mask

        # Transformer Encoder üzerinden geçirme
        transformer_out = self.transformer_encoder(padded_x, src_key_padding_mask=src_key_padding_mask)

        # Global mean pooling: geçerli düğümlerin ortalaması alınır.
        transformer_out = transformer_out * mask.unsqueeze(-1)  # mask'i genişleterek çarpıyoruz
        graph_rep = transformer_out.sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # Fully-connected katman ile sınıflandırma
        out = self.fc(graph_rep)
        return out