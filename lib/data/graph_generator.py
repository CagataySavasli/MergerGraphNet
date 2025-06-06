import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

class GraphGenerator(Dataset):
    def __init__(self, num_conn: int=2):
        self.num_conn = num_conn

    def generate_graph(self, embeddings: list[list[float]]):

        similarities = cosine_similarity(embeddings)
        threashold = similarities.mean()


        x = torch.stack(
            [
                torch.as_tensor(vec, dtype=torch.float32).clone().detach()
                for vec in embeddings
            ],
            dim=0
        ).clone().detach()

        edge_list = []
        edge_attr = []

        for node_idx in range(len(embeddings)):
            end_idx = node_idx + self.num_conn
            while end_idx > len(embeddings):
                end_idx -= 1
            if end_idx == len(embeddings):
                break
            for conn_idx in range(node_idx+1, end_idx+1):
                edge_list.append((node_idx, conn_idx))

                similarity = similarities[node_idx, conn_idx]
                distance = 1/(abs(node_idx-conn_idx))

                edge_attr.append((distance, similarity))

        for node_idx in range(len(embeddings)):
            for conn_idx in range(node_idx+1, len(embeddings)):
                pear = (node_idx, conn_idx)
                if abs(node_idx-conn_idx) < self.num_conn or node_idx != conn_idx:

                    similarity = similarities[node_idx, conn_idx]
                    distance = 1/(abs(node_idx-conn_idx))

                    if similarity > threashold:
                        edge_list.append(pear)
                        edge_attr.append((distance, similarity))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)

        # if not edge_index.any():
        #     print("Empty edge list")
        # if not edge_attr.any():
        #     print("Empty edge list")

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


