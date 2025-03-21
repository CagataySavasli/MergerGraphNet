from lib.util.graph_generator import GraphGenerator
from sklearn.metrics.pairwise import cosine_similarity
from lib.config.config_loader import ConfigLoader

class GraphDataLoader(torch.utils.data.Dataset):
    def __init__(self, df, n):
        self.config = ConfigLoader().load_config()
        self.generator = GraphGenerator()
        self.df = df
        self.n = n

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.generator(row)