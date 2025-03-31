"""
This module defines the `GraphDataLoader` class, which leverages a `GraphGenerator`
to produce PyTorch Geometric `Data` objects from a DataFrame. Each row in the DataFrame
represents an instance with sentence embeddings and a label, and the `GraphGenerator`
converts these embeddings into a graph structure.
"""

import torch
from torch.utils.data import Dataset
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity  # Not currently used, but available if needed.
from lib.util.graph_generator import GraphGenerator
from lib.config.config_loader import ConfigLoader


class GraphDataLoader(Dataset):
    """
    A custom PyTorch Dataset that transforms rows from a DataFrame into
    graph data using the `GraphGenerator`.
    """

    def __init__(self, df, n: int) -> None:
        """
        Initialize the GraphDataLoader.

        Args:
            df (pd.DataFrame): A DataFrame with at least two columns:
                               'embeddings' (list/array of sentence vectors)
                               and 'label' (int).
            n (int): The window size for connecting sentences in the graph.
        """
        super().__init__()
        self._config = ConfigLoader().config
        self._df = df
        self._n = n
        # Pass n to the GraphGenerator so it can use this for window-based edges.
        self._generator = GraphGenerator(n=self._n)

    def __len__(self) -> int:
        """
        Return the total number of rows in the DataFrame.

        Returns:
            int: The length of the dataset.
        """
        return len(self._df)

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve a single item (graph data and label) from the DataFrame at the specified index.

        Args:
            idx (int): Index of the row to retrieve from the DataFrame.

        Returns:
            Any: A tuple (Data, torch.Tensor) as returned by the GraphGenerator, 
                 where `Data` is a PyTorch Geometric `Data` object and the second item is the label tensor.
        """
        row = self._df.iloc[idx]
        return self._generator(row)
