from lib.util.graph_generator import GraphGenerator
import pandas as pd
import numpy as np


class GraphIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_file, chunk_size=1000, n=10):
        """
        Args:
            csv_file (str): İşlenecek CSV dosyasının yolu.
            chunk_size (int): CSV dosyasını okurken kullanılacak satır adedi.
            n (int): Komşu pencere boyutu (her cümle için n komşuya kenar oluşturuluyor).
        """
        self.csv_file = csv_file
        self.graph_generator = GraphGenerator()
        self.chunk_size = chunk_size
        self.n = n

    def parse_row(self, row):
        # CSV'de 'embeddings' sütunu string olarak saklanmışsa, eval ile dönüştürüyoruz.
        env = {"array": np.array, "float16": np.float16}
        try:
            row['embeddings'] = eval(row['embeddings'], env)
        except Exception:
            row['embeddings'] = row['embeddings']

        return self.graph_generator(row)



    def __iter__(self):
        # CSV'yi belirli boyutlarda (chunk_size) okuyarak her satır için örnek üretiyoruz.
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                yield self.parse_row(row)
