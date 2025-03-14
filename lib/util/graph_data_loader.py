import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from lib.config.config_loader import ConfigLoader

class GraphDataLoader(torch.utils.data.Dataset):
    def __init__(self, df, n):
        self.config = ConfigLoader().load_config()
        self.df = df
        self.n = n

    def get_graph(self, row):
        # TF-IDF vektörlerini alıyoruz. Eğer sparse ise dense formata çeviriyoruz.
        sentence_vectors = row['tfidf_sentence']
        sentence_vectors = sentence_vectors.toarray()

        # Cümleler arası cosine similarity hesaplanıyor.
        similarity_matrix = cosine_similarity(sentence_vectors)

        # Cümle sayısını, mda sütunundaki cümle listesine göre belirliyoruz.
        num_sentences = len(row['sentences'])
        label =torch.tensor([row['label']])


        # Her cümlenin TF-IDF vektörünü torch tensor'a dönüştürüyoruz.
        x = torch.stack(
            [torch.tensor(sentence_vectors[i], dtype=torch.float) for i in range(num_sentences)],
            dim=0
        )

        # Kenar listesi ve kenar ağırlıklarını (edge_attr) oluşturuyoruz.
        edge_list = []
        edge_attr = []
        for i in range(num_sentences):
            for j in range(max(0, i - self.n), min(num_sentences, i + self.n + 1)):
                if i != j:
                    edge_list.append([i, j])
                    edge_attr.append(similarity_matrix[i][j])

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # PyTorch Geometric Data objesi oluşturuluyor.
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return self.get_graph(row)