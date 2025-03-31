"""
This module defines the `EmbeddingDataLoader` class, which handles the loading
of sentence embeddings and labels from a Pandas DataFrame. It integrates
seamlessly with PyTorch's DataLoader, enabling easy batching and sampling
for training or inference workflows.
"""

import torch
from torch.utils.data import Dataset
from typing import Any, Tuple
from lib.config.config_loader import ConfigLoader


class EmbeddingDataLoader(Dataset):
    """
    A custom PyTorch Dataset for loading embeddings and labels from a DataFrame.
    """

    def __init__(self, df) -> None:
        """
        Initializes the dataset using a DataFrame containing embeddings and labels.

        Args:
            df (pd.DataFrame): A DataFrame that must contain 'embeddings' and 'label' columns.
        """
        self._config = ConfigLoader().config
        self._df = df.reset_index(drop=True)
        # This attribute is read from the config, but not used in this code snippet.
        # Retain if needed for future logic.
        self._n = self._config['data_loader'].get('graph_n_sentence_connection', None)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset at the specified index.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the embeddings (data) and the label.
        """
        row = self._df.iloc[idx]
        return self._process_row(row)

    def _process_row(self, row: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single row from the DataFrame to extract embeddings and convert them to Tensors.

        Args:
            row (Any): A row from the DataFrame that includes 'embeddings' and 'label' fields.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the embeddings tensor and the label tensor.
        """
        sentence_vectors = row['embeddings']
        num_sentences = sentence_vectors.shape[0]

        # Convert label to a tensor (int) with shape [1].
        label = torch.tensor([row['label']], dtype=torch.long)

        # Convert each embedding to a float tensor, then stack them
        # into one tensor with shape [num_sentences, embedding_dim].
        data = torch.stack(
            [torch.tensor(sentence_vectors[i], dtype=torch.float32) for i in range(num_sentences)],
            dim=0
        )

        return data, label
