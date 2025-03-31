"""
This script generates sentence embeddings for a subset of rows in a CSV dataset
using a pre-trained Transformer model. It relies on a configuration file for
model and I/O paths. Usage:

    python embed_sentences.py <start_index> <end_index>

The script reads the CSV, tokenizes the 'mda' column into sentences, obtains
embeddings, and saves the results as a Parquet file.

Main Steps:
1. Load configuration and model tokenizer and weights.
2. Determine the subset of the dataset to process using command-line arguments.
3. Tokenize the 'mda' text into sentences (using NLTK's sent_tokenize).
4. Generate sentence embeddings in batches to manage memory usage.
5. Save the final DataFrame as a Parquet file.
"""

import sys

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel

from lib.config.config_loader import ConfigLoader

print("Loading model...")

config = ConfigLoader().config
tqdm.pandas()

# Load the model and tokenizer from the config.
tokenizer = AutoTokenizer.from_pretrained(config['embedding']['model_name'])
model = AutoModel.from_pretrained(config['embedding']['model_name'])

# Parse command-line arguments to get start and end indices.
start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

print(f"Preprocessing rows {start_idx} to {end_idx}...")
df = pd.read_csv('./data/processed/reports_labeled.csv')
df = df.loc[start_idx:end_idx].copy()
df.reset_index(drop=True, inplace=True)

# Split the 'mda' text into sentences.
df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))
df.drop('mda', axis=1, inplace=True)


def get_finbert_embedding(texts, batch_size: int = 256):
    """
    Generate embeddings for a list of texts using the pre-trained model in batches.
    If a memory error occurs, recursively reduce the batch size until a lower limit is reached.

    Args:
        texts (List[str]): A list of sentences or text passages to embed.
        batch_size (int): The number of texts to process per batch. Defaults to 256.

    Returns:
        List[numpy.ndarray]: A list of embeddings corresponding to the input texts.

    Raises:
        RuntimeError: If an error occurs at batch size = 2 (lowest fallback).
    """
    embeddings = []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                outputs = model(**tokens)
                # Convert embeddings to FP16 and move to CPU for final numpy conversion.
                batch_embeddings = outputs.pooler_output.half().cpu().numpy()
                embeddings.extend(batch_embeddings)
    except Exception as e:
        # If a memory or another error occurs, try reducing the batch size.
        if batch_size > 2:
            return get_finbert_embedding(texts, batch_size=batch_size // 2)
        else:
            raise RuntimeError(f"Error during embedding generation: {e}")
    return embeddings


print("Generating embeddings...")
# For each row, compute embeddings for all sentences in 'sentences'.
df['embedding'] = df['sentences'].progress_apply(get_finbert_embedding)

# Save results in Parquet format for efficient disk usage and fast I/O.
output_file = f'./data/sep/embeddings_labeled_{start_idx}_{end_idx}.parquet'
print(f"Saving embeddings to {output_file}...")
df.to_parquet(output_file, index=False)

print("Done!")
