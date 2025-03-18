import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from lib.config.config_loader import ConfigLoader


# Custom dataset sınıfı: Her çağrıda ilgili satır için graph oluşturulur.
class EmbeddingDataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.config = ConfigLoader().load_config()
        self.df = df.reset_index(drop=True)
        self.n = self.config['data_loader']['graph_n_sentence_connection']

    def get_sentences(self, row):
        sentence_vectors = row['tfidf_sentence']
        sentence_vectors = sentence_vectors.toarray()  # Dense array'e çevirme
        num_sentences = sentence_vectors.shape[0]
        label = torch.tensor([row['label']])

        data = torch.stack(
            [torch.tensor(sentence_vectors[i], dtype=torch.float) for i in range(num_sentences)],
            dim=0
        )
        return data, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return self.get_sentences(row)
