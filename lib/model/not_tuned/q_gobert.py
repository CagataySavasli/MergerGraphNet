import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GlobalAttention

class Q_GoBERT(nn.Module):
    def __init__(self, input_dim, quant_input_dim, hidden_dim_1, hidden_dim_2, lstm_hidden_dim, output_dim=1, dropout=0.5):
        super(Q_GoBERT, self).__init__()

        # Graph BERT layers
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1, heads=4, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1 * 4)

        self.conv2 = GCNConv(hidden_dim_1 * 4, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)

        self.att_pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim_2, 1)))
        self.dropout = nn.Dropout(dropout)

        # LSTM for quant input
        self.quant_lstm = nn.LSTM(
            input_size=quant_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Combined classifier
        self.fc = nn.Linear(hidden_dim_2 + lstm_hidden_dim, output_dim)

    def forward(self, data, quantative_data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        graph_rep = self.att_pool(x, batch)  # [batch_size, hidden_dim_2]

        # Quantitative LSTM
        _, (h_n_quant, _) = self.quant_lstm(quantative_data)  # [1, batch_size, lstm_hidden_dim]
        quant_hidden = h_n_quant[-1]  # [batch_size, lstm_hidden_dim]

        # Concatenate and classify
        combined = torch.cat([graph_rep, quant_hidden], dim=1)
        out = self.fc(combined).squeeze(1)
        return out
