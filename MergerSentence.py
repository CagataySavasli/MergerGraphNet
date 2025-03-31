"""
This script trains a sentence-level classifier on financial text data (the 'mda' column).
Sentences are embedded using TF-IDF, then collated and passed to either an LSTM- or
Transformer-based classifier, based on the command-line argument (`strategy`).

Where <strategy> is either:
    - "LSTM": Uses an LSTM-based model (SentenceLSTMClassifier).
    - "SentenceTrans": Uses a Transformer-based model (SentenceTransClassifier).

Workflow Steps:
1. Load configuration from a YAML file for model hyperparameters (e.g., input_dim).
2. Read a CSV file (`reports_labeled.csv`), tokenize the 'mda' text into sentences.
3. Split the dataset into training (year <= 2019) and test (year > 2019).
4. Compute TF-IDF embeddings for each sentence using a max feature size from the config.
5. Use a custom DataLoader (EmbeddingDataLoader) and a custom collate function to batch and pad sequences.
6. Instantiate the specified model (LSTM or SentenceTrans) and train for a fixed number of epochs.
7. Track loss and evaluate the model each epoch on the test set (accuracy, precision, recall, F1, confusion matrix).
8. Save final results to a CSV file in `./outputs/`.

Note:
- This script is CPU-based by default (device='cpu'), but you can enable CUDA if available.
- Hyperparameters are partially read from the YAML config.
"""

# %% [Imports]
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Local Imports
from lib.util.embedding_data_loader import EmbeddingDataLoader
from lib.model.sentence_classifer import SentenceLSTMClassifier, SentenceTransClassifier
from lib.config.config_loader import ConfigLoader

# %% [Global Configuration & Device]
device = torch.device("cpu")  # Switch to torch.device("cuda") if GPU is available.

# Load YAML config
config = ConfigLoader().config
tqdm.pandas()

# %% [Command-Line Argument for Model Strategy]
strategy = sys.argv[1]  # "LSTM" or "SentenceTrans"

# %% [Hyperparameters from Config]
input_dim = config['models']['input_dim']
hidden_dim_1 = config['models']['hidden_dim_1']
hidden_dim_2 = config['models']['hidden_dim_2']
hidden_dim_3 = config['models']['hidden_dim_3']

# %% [Data Preprocessing]
print("Reading and preprocessing data...")

df = pd.read_csv('./data/processed/reports_labeled.csv')
df.reset_index(drop=True, inplace=True)

# Tokenize each report into sentences
df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))

# Split train/test by year
train_df = df[df['year'] <= 2019].copy().reset_index(drop=True)
test_df = df[df['year'] > 2019].copy().reset_index(drop=True)

# Class weighting for imbalanced data
merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 1]['label'])
not_merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 0]['label'])
class_weights = torch.tensor([merge_class_weight, not_merge_class_weight], dtype=torch.float).to(device)

print(f"""
## Length of dataset:
Length of training set: {len(train_df)}
Length of test set: {len(test_df)}

## Distribution of label (Training):
Number of 'merge' (label=1): {len(train_df[train_df['label'] == 1])}
Number of 'not merge' (label=0): {len(train_df[train_df['label'] == 0])}
Class weights: {class_weights}

Number of 'merge' in test set: {len(test_df[test_df['label'] == 1])}
Number of 'not merge' in test set: {len(test_df[test_df['label'] == 0])}
""")

# Fit a TF-IDF vectorizer on all training sentences
train_corpus = [sent for sents in train_df['sentences'] for sent in sents]
vectorizer = TfidfVectorizer(max_features=input_dim, stop_words='english')
vectorizer.fit(train_corpus)


def get_tfidf_embeddings(sentence_list):
    """
    Converts a list of sentences to TF-IDF embeddings (sparse matrices).

    Args:
        sentence_list (List[str]): A list of raw sentences.

    Returns:
        scipy.sparse.csr_matrix: Sparse TF-IDF vector for each sentence.
    """
    if not isinstance(sentence_list, list):
        sentence_list = [sentence_list]
    return vectorizer.transform(sentence_list)


print("Generating TF-IDF embeddings for train and test sets...")
train_df['tfidf_sentence'] = train_df['sentences'].progress_apply(get_tfidf_embeddings)
test_df['tfidf_sentence'] = test_df['sentences'].progress_apply(get_tfidf_embeddings)

# %% [Dataset & DataLoader]
print("Creating EmbeddingDataLoader instances...")

train_dataset = EmbeddingDataLoader(train_df)
test_dataset = EmbeddingDataLoader(test_df)


def custom_collate(batch):
    """
    Custom collate function for variable-length sentence embeddings.
    Batches a list of (data, label) pairs from EmbeddingDataLoader and pads
    the sentence embeddings along the time (sentence) dimension.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A batch of data returned by the Dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded data of shape
            (batch_size, max_num_sentences, input_dim),
            sequence lengths for each sample (batch_size),
            and the labels (batch_size, 1).
    """
    data_list, label_list = zip(*batch)
    lengths = torch.tensor([data.size(0) for data in data_list])

    # pad_sequence => shape [batch_size, max_time, embedding_dim] when batch_first=True.
    padded_data = pad_sequence(data_list, batch_first=True)
    labels = torch.stack(label_list, dim=0)

    return padded_data, lengths, labels


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)


# %% [Evaluation Function]
def evaluate(y_true, y_pred):
    """
    Compute common classification metrics and confusion matrix elements.

    Args:
        y_true (List[int]): Ground-truth labels.
        y_pred (List[int]): Predicted labels by the model.

    Returns:
        (accuracy, precision, recall, f1, tp, tn, fp, fn)
    """
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return accuracy, precision, recall, f1, tp, tn, fp, fn


# %% [Train & Test Functions]
def train_one_epoch():
    """
    Train the model for one epoch over the training DataLoader.

    Returns:
        float: The average training loss.
    """
    model.train()
    total_loss = 0.0
    for data, lengths, label in tqdm(train_loader, desc="Training", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data, lengths)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def test_model(loader):
    """
    Evaluate the model on a given DataLoader.

    Args:
        loader (DataLoader): Loader for validation/test data.

    Returns:
        Tuple[List[int], List[int]]: Ground truth labels and model predictions.
    """
    model.eval()
    y_pred, y_true = [], []

    for data, lengths, label in tqdm(loader, desc="Testing", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)
        with torch.no_grad():
            out = model(data, lengths)
            pred = out.argmax(dim=1)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return y_true, y_pred


# %% [Initialize Model]
print(f"Selected strategy: {strategy}")
if strategy == "LSTM":
    model = SentenceLSTMClassifier(
        input_dim=input_dim,
        fc1_dim=hidden_dim_1,
        fc2_dim=hidden_dim_2,
        lstm_hidden_dim=hidden_dim_3,
        output_dim=2
    ).to(device)
    model_name = "SentenceClassifierLSTM"
elif strategy == "SentenceTrans":
    model = SentenceTransClassifier(
        input_dim=input_dim,
        fc1_dim=hidden_dim_1,
        fc2_dim=hidden_dim_2,
        transformer_hidden_dim=hidden_dim_3,
        output_dim=2
    ).to(device)
    model_name = "SentenceClassifierTransformer"
else:
    raise ValueError(f"Unknown strategy: {strategy}")

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# %% [Training Loop]
print("Starting Training...")
num_epochs = 15
for epoch in range(1, num_epochs + 1):
    loss_value = train_one_epoch()
    y_true_test, y_pred_test = test_model(test_loader)

    accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(y_true_test, y_pred_test)
    print(f"Epoch: {epoch:02d}, Loss: {loss_value:.4f} | "
          f"Test => Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1} | "
          f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("Training complete.")

# %% [Save Results]
print("Saving final results...")
result_dict = {
    "Model": [model_name],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "TP": [tp],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn]
}
result_df = pd.DataFrame(data=result_dict)

output_filename = f'./outputs/{model_name}_results.csv'
result_df.to_csv(output_filename, index=False)

print(f"Done! Results saved to {output_filename}")
