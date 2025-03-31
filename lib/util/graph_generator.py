"""
This module defines the `GraphGenerator` class, which converts row-based text
embeddings into a PyTorch Geometric Data object. By default, it constructs edges
using a sliding window determined by the `n` parameter and assigns edge weights
based on the distance between sentence indices (1 / |i-j|). You can modify the
edge logic (for example, by using cosine similarity) if needed.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Any, Tuple


class GraphGenerator(Dataset):
    """
    A PyTorch-compatible generator that transforms sentence embeddings from
    a single data row into a graph structure. Edges are created around each
    sentence index within a configurable window defined by `n`.
    """

    def __init__(self, n: int = 5) -> None:
        """
        Initialize the GraphGenerator.

        Args:
            n (int): The window size for connecting sentences. For each sentence
                     at index i, edges will be created to sentences in
                     [i - n, i + n], excluding i.
        """
        super().__init__()
        self.n = n

    def _get_graph(self, row: Any) -> Tuple[Data, torch.Tensor]:
        """
        Convert a single data row into a PyTorch Geometric Data object and a label tensor.

        Args:
            row (Any): A data row which should contain:
                - 'embeddings': list/array of sentence vectors
                - 'label': integer label for classification or other tasks

        Returns:
            Tuple[Data, torch.Tensor]:
                - Data: A PyTorch Geometric Data object that includes:
                    * x (torch.Tensor): Node features of shape [num_sentences, embedding_dim].
                    * edge_index (torch.Tensor): Graph edges of shape [2, E].
                    * edge_attr (torch.Tensor): Edge weights of shape [E, 1].
                - torch.Tensor: Label tensor of shape [1].
        """
        sentence_vectors = row["embeddings"]
        num_sentences = len(sentence_vectors)

        # Convert label to a single-element tensor.
        label = torch.tensor([row["label"]], dtype=torch.long)

        # Stack each embedding into a float32 tensor with shape [num_sentences, embedding_dim].
        x = torch.stack(
            [torch.tensor(vec, dtype=torch.float32) for vec in sentence_vectors],
            dim=0
        )

        # Create edges using a sliding window approach (range: [i-n, i+n]).
        edge_list = []
        edge_attr = []
        for i in range(num_sentences):
            start_idx = max(0, i - self.n)
            end_idx = min(num_sentences, i + self.n + 1)
            for j in range(start_idx, end_idx):
                if i != j:
                    edge_list.append([i, j])
                    # Here we assign a simple distance-based weight. If you'd like to use
                    # cosine similarity, you can compute it here instead of 1 / abs(i-j).
                    edge_attr.append(1 / abs(i - j))

        # Convert edge data to PyTorch tensors.
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        # Construct the PyTorch Geometric Data object.
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data, label

    def __call__(self, row: Any) -> Tuple[Data, torch.Tensor]:
        """
        Make the class instance callable, generating a graph from the given row.

        Args:
            row (Any): A data row containing 'embeddings' and 'label'.

        Returns:
            Tuple[Data, torch.Tensor]: The generated graph data and the label tensor.
        """
        return self._get_graph(row)
