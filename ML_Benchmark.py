"""
This script demonstrates a simple machine learning pipeline for binary classification
using multiple classifiers (GaussianNB, LogisticRegression, RandomForestClassifier, and XGBClassifier).
The text data comes from a CSV file (reports_labeled.csv) containing an 'mda' column (text) and a
'binary' label column. TF-IDF embeddings are generated for each record, and then training and
evaluation are performed.
"""

# %% [Imports]
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tqdm import tqdm

# Local import
from lib.config.config_loader import ConfigLoader


# Load the configuration and set up progress bar
config = ConfigLoader().config
tqdm.pandas()

# Define TF-IDF dimensionality
input_dim = config['models']['input_dim']

print("Reading CSV data...")
df = pd.read_csv('./data/processed/reports_labeled.csv')
df.reset_index(drop=True, inplace=True)

# Tokenize 'mda' text into sentences (though we mainly use the whole text for embeddings)
df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))

# Split into train/test based on the 'year' column
train_df = df[df['year'] <= 2019].copy()
test_df = df[df['year'] > 2019].copy()

print("Fitting TF-IDF vectorizer on training data (all sentences)...")
train_corpus = [
    sentence
    for sentences in train_df['sentences']
    for sentence in sentences
]
vectorizer = TfidfVectorizer(max_features=input_dim, stop_words='english')
vectorizer.fit(train_corpus)

def get_tfidf_embeddings(text: str):
    """
    Convert a given text string (or list of strings) into TF-IDF embeddings.

    Args:
        text (str or List[str]): The text to transform into TF-IDF vectors.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix containing the TF-IDF representation.
    """
    if not isinstance(text, list):
        text = [text]
    return vectorizer.transform(text)

print("Transforming training data to TF-IDF...")
train_df['tfidf_mda'] = train_df['mda'].progress_apply(get_tfidf_embeddings)
print("Transforming testing data to TF-IDF...")
test_df['tfidf_mda'] = test_df['mda'].progress_apply(get_tfidf_embeddings)

# Prepare final train and test sets
X_train = [row.toarray()[0] for row in train_df['tfidf_mda']]
y_train = train_df['label'].to_numpy()

X_test = [row.toarray()[0] for row in test_df['tfidf_mda']]
y_test = test_df['label'].to_numpy()

# Define the models to train
models = {
    "Gaussian NB": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

# Train and store predictions
predictions = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = {'y_pred': y_pred, 'y_test': y_test}

# Evaluate each model
def evaluate(y_true, y_pred):
    """
    Calculate and return classification metrics.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels by the model.

    Returns:
        Tuple[float, float, float, float, int, int, int, int]:
            accuracy, precision, recall, f1, tp, tn, fp, fn
    """
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return accuracy, precision, recall, f1, tp, tn, fp, fn

# Generate results dictionary
results = {
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

# Evaluate each model and collect metrics
print("Evaluating all models...")
for name, pred_dict in predictions.items():
    print(f"Evaluating {name}...")
    y_true, y_pred = pred_dict['y_test'], pred_dict['y_pred']
    accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(y_true, y_pred)

    results["Model"].append(name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)
    results["TP"].append(tp)
    results["TN"].append(tn)
    results["FP"].append(fp)
    results["FN"].append(fn)

    print(
        f"{name} => Accuracy: {accuracy}, Precision: {precision}, "
        f"Recall: {recall}, F1: {f1} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
    )
    print("#" * 100)

# Convert results to a DataFrame and save
results_df = pd.DataFrame(results)
output_path = './outputs/ml_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Done! Results saved to {output_path}")
