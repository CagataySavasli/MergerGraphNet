import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GlobalAttention

class GoBERT(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout=0.5):
        super(GoBERT, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim_1, heads=4, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1 * 4)

        self.conv2 = GCNConv(hidden_dim_1 * 4, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)

        self.att_pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(hidden_dim_2, 1)))

        self.fc = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        graph_rep = self.att_pool(x, batch)

        # Final sınıflandırma katmanı
        out = self.fc(graph_rep)
        out = out.squeeze(1)
        return out
