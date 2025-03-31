"""
This script trains a graph-based classification model using PyTorch Geometric
and a user-configurable embedding approach (TF-IDF or BERT embeddings). It
relies on configuration settings from a YAML file and takes command-line
arguments to specify the model type and embedding type.

Usage Example:
    python train_graph_model.py GraphClassifier tfidf

Where:
    - The first argument (e.g., 'GraphClassifier') is one of the supported model names:
        * GraphClassifier
        * GraphResidualClassifier
        * GraphLSTMClassifier
        * GraphTransClassifier
    - The second argument (e.g., 'tfidf') indicates the embedding type:
        * 'tfidf' uses TF-IDF embeddings
        * 'bert' (or any other approach) is handled via a separate dataset loader.

Functionality:
1. Loads configuration (hyperparameters, paths, etc.) from a YAML file.
2. Reads a labeled dataset (CSV with 'mda' text and a binary 'label').
3. If TF-IDF is specified, transforms text data to embeddings using TfidfVectorizer.
4. If BERT is specified, uses a different approach (GraphIterableDataset) to generate embeddings.
5. Constructs a graph for each example using a pre-defined window size ('n').
6. Trains a specified graph-based model (such as GraphResidualClassifier) using PyTorch.
7. Evaluates the model on a held-out test set and reports common classification metrics.
8. Logs progress via `tqdm` during both training and testing phases.
9. Saves evaluation results to CSV.

Note:
- This script is CPU-based by default (device='cpu'), but you can enable CUDA if available.
- Hyperparameters (batch_size, window size, input dimension, etc.) are partially read from the YAML config.

Author: Your Name
Date: 2025-03-31
"""

# %% [Imports]
import sys
import ast

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from nltk.tokenize import sent_tokenize
from torch_geometric.loader import DataLoader

# Local Imports
from lib.util.graph_data_loader import GraphDataLoader
from lib.util.graph_iterable_data_loader import GraphIterableDataset
from lib.model.graph_classifer import (
    GraphClassifier,
    GraphResidualClassifier,
    GraphLSTMClassifier,
    GraphTransClassifier
)
from lib.config.config_loader import ConfigLoader

# %% [Device Configuration]
device = torch.device("cpu")  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [Load Config]
config = ConfigLoader().config
tqdm.pandas()

# %% [Hyperparameters]
input_dim = config['models']['input_dim']
hidden_dim_1 = config['models']['hidden_dim_1']
hidden_dim_2 = config['models']['hidden_dim_2']
hidden_dim_3 = config['models']['hidden_dim_3']

# %% [CLI Arguments]
model_name = sys.argv[1]  # e.g., 'GraphResidualClassifier'
embedding_type = sys.argv[2]  # e.g., 'tfidf'

# %% [Data Loading & Preprocessing]
print("Start Preprocess")
if embedding_type == 'tfidf':
    # Read the CSV with text data
    df = pd.read_csv('./data/processed/reports_labeled.csv')
    df.reset_index(drop=True, inplace=True)

    # Split 'mda' text into sentences
    df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))

    # Separate into train/test based on 'year'
    train_df = df[df['year'] <= 2019].copy().reset_index(drop=True)
    test_df = df[df['year'] > 2019].copy().reset_index(drop=True)

    # Calculate class weights for imbalanced data
    merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 1]['label'])
    not_merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 0]['label'])
    class_weights = torch.tensor([merge_class_weight, not_merge_class_weight], dtype=torch.float).to(device)

    print(f"""
    ## Length of dataset:
    Length of training set: {len(train_df)}
    Length of test set: {len(test_df)}

    ## Distribution of label:
    Number of merge in training set: {len(train_df[train_df['label'] == 1]['label'])}
    Number of not-merge in training set: {len(train_df[train_df['label'] == 0]['label'])}
    Class weights: {class_weights}

    Number of merge in test set: {len(test_df[test_df['label'] == 1]['label'])}
    Number of not-merge in test set: {len(test_df[test_df['label'] == 0]['label'])}
    """)


    def get_tfidf_embeddings(sentence_list):
        """
        Converts a list of sentences into TF-IDF vectors using a pre-fitted TfidfVectorizer.

        Args:
            sentence_list (List[str]): A list of raw sentences.

        Returns:
            np.ndarray: An array of shape [num_sentences, input_dim] representing TF-IDF embeddings.
        """
        if not isinstance(sentence_list, list):
            sentence_list = [sentence_list]
        embeddings = vectorizer.transform(sentence_list)
        return embeddings.toarray()


    # Prepare TF-IDF vectorizer on all training sentences
    train_corpus = [s for sentences in train_df['sentences'] for s in sentences]
    vectorizer = TfidfVectorizer(max_features=input_dim, stop_words='english')
    vectorizer.fit(train_corpus)

    print("Train Sentence: ")
    train_df['embeddings'] = train_df['sentences'].progress_apply(get_tfidf_embeddings)

    print("Test Sentence: ")
    test_df['embeddings'] = test_df['sentences'].progress_apply(get_tfidf_embeddings)
    print("End Getting Tfidf Embeddings")

    # Create graph data sets
    train_dataset = GraphDataLoader(train_df, n=10)
    test_dataset = GraphDataLoader(test_df, n=10)

else:
    print("BERT will be used for embedding!!")

    # Hardcoded class weights for demonstration; adjust as needed.
    class_weights = torch.tensor([4.488888888888889, 1.286624203821656], dtype=torch.float).to(device)

    # CSV files that contain pre-embedded BERT representations
    train_csv = 'data/processed/embedded_labeled_train.csv'
    test_csv = 'data/processed/embedded_labeled_test.csv'

    train_dataset = GraphIterableDataset(train_csv, chunk_size=128, n=10)
    test_dataset = GraphIterableDataset(test_csv, chunk_size=128, n=10)

# %% [DataLoader Creation]
train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


# %% [Evaluation Helper Function]
def evaluate(y_true, y_pred):
    """
    Compute classification metrics (accuracy, precision, recall, F1) and confusion matrix elements.

    Args:
        y_true (List[int]): Ground-truth labels.
        y_pred (List[int]): Predicted labels.

    Returns:
        Tuple[float, float, float, float, int, int, int, int]:
            accuracy, precision, recall, f1, TP, TN, FP, FN
    """
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return accuracy, precision, recall, f1, tp, tn, fp, fn


# %% [Training Function]
def train():
    """
    Train the model for one epoch over the training DataLoader.

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    total_loss = 0
    for data, label in tqdm(train_loader, desc="Training", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)  # Model output (logits)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


# %% [Testing Function]
def test(loader):
    """
    Evaluate the model on a given DataLoader, returning the predictions and ground truths.

    Args:
        loader (DataLoader): DataLoader for validation or test data.

    Returns:
        Tuple[List[int], List[int]]: The ground truth (y_true) and predictions (y_pred).
    """
    model.eval()
    y_pred = []
    y_true = []

    for data, label in tqdm(loader, desc="Testing", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return y_true, y_pred


# %% [Model Initialization & Training]
result_dict = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "TP": [],
    "TN": [],
    "FP": [],
    "FN": []
}

print("Training Model:", model_name)

# Map the model_name argument to a specific model class
if model_name == 'GraphClassifier':
    model = GraphClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        output_dim=2
    ).to(device)
elif model_name == 'GraphLSTMClassifier':
    model = GraphLSTMClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        lstm_hidden_dim=hidden_dim_3,
        output_dim=2
    ).to(device)
elif model_name == 'GraphTransClassifier':
    model = GraphTransClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        nhead=4,
        num_encoder_layers=2,
        output_dim=2
    ).to(device)
elif model_name == 'GraphResidualClassifier':
    model = GraphResidualClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        hidden_dim_3=hidden_dim_3,
        output_dim=2
    ).to(device)
else:
    raise ValueError(f"Unsupported model name: {model_name}")

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Start Training")
num_epochs = 15
for epoch in range(1, num_epochs + 1):
    loss = train()

    # Evaluate on the test set each epoch
    y_true_test, y_pred_test = test(test_loader)
    accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(y_true_test, y_pred_test)

    if epoch % 3 == 0 or epoch == num_epochs or epoch == 1:
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f} | Test | "
              f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1} | "
              f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("End Training")

# Save final metrics
result_dict['Model'].append(model_name)
result_dict['Accuracy'].append(accuracy)
result_dict['Precision'].append(precision)
result_dict['Recall'].append(recall)
result_dict['F1'].append(f1)
result_dict['TP'].append(tp)
result_dict['TN'].append(tn)
result_dict['FP'].append(fp)
result_dict['FN'].append(fn)

result_df = pd.DataFrame(data=result_dict)
print("Results:")
print(result_df)

output_csv = f'./outputs/{model_name}_{embedding_type}_results.csv'
result_df.to_csv(output_csv, index=False)
print(f"Done! Results saved to {output_csv}")
