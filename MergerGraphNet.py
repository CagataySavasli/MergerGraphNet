#%%
from nltk.tokenize import sent_tokenize

from lib.util.graph_data_loader import GraphDataLoader
from lib.model.graph_classifer import GraphClassifier, GraphResidualClassifier, GraphLSTMClassifier, GraphTransClassifier
from lib.config.config_loader import ConfigLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

config = ConfigLoader().load_config()
tqdm.pandas()
#%%
input_dim = config['models']['input_dim']
hidden_dim_1 = config['models']['hidden_dim_1']
hidden_dim_2 = config['models']['hidden_dim_2']
hidden_dim_3 = config['models']['hidden_dim_3']

model_name = sys.argv[1]#'GraphResidualClassifier'
embedding_type = sys.argv[2]#'tfidf'
#%%
print("Start Preprocess")
if embedding_type == 'tfidf':
    df = pd.read_csv('./data/processed/reports_labeled.csv')

    df.reset_index(drop=True, inplace=True)
    df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))

else:
    df = pd.read_csv('./data/processed/embedded_labeled.csv')


train_df = df[df['year'] <= 2019].copy().reset_index(drop=True)
test_df = df[df['year'] > 2019].copy().reset_index(drop=True)

merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 1]['label'])
not_merge_class_weight = len(train_df) / len(train_df[train_df['label'] == 0]['label'])

class_weights = torch.tensor([merge_class_weight, not_merge_class_weight], dtype=torch.float).to(device)
#%%
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
#%%
if embedding_type == 'tfidf':
    train_corpus = [sentence for sentences in train_df['sentences'] for sentence in sentences]

    vectorizer = TfidfVectorizer(max_features=input_dim, stop_words='english')
    vectorizer.fit(train_corpus)

    def get_tfidf_embeddings(sentence_list):
        if not type(sentence_list) == list:
            sentence_list = [sentence_list]
        embeddings = vectorizer.transform(sentence_list)
        return embeddings

    print("Train Sentence: ")
    train_df['embeddings'] = train_df['sentences'].progress_apply(get_tfidf_embeddings)
    print("Test Sentence: ")
    test_df['embeddings'] = test_df['sentences'].progress_apply(get_tfidf_embeddings)
    print("End Getting Tfidf Embeddings")
#%%
train_dataset = GraphDataLoader(train_df, 10)
test_dataset = GraphDataLoader(test_df, 10)
#%%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#%%
def evaluate(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return accuracy, precision, recall, f1, tp, tn, fp, fn

def train():
    model.train()
    total_loss = 0
    # train_loader üzerinden geçerken progress bar ekleniyor.
    for data, label in tqdm(train_loader, desc="Training", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # Modelin çıktısı (logits)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Test fonksiyonu
def test(loader):
    model.eval()
    correct = 0

    y_pred = []
    y_true = []

    for data, label in tqdm(loader, desc="Testing", leave=False):
        label = label.squeeze_(1).to(device)
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)  # En yüksek logit değerine sahip sınıf

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return y_true, y_pred
#%%
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

print("Training Model: ", model_name)
# Model, optimizer ve loss function tanımlanıyor

if model_name == 'GraphClassifier':
    model = GraphClassifier(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, output_dim=2).to(device)
elif model_name == 'GraphLSTMClassifier':
    model = GraphLSTMClassifier(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, lstm_hidden_dim=hidden_dim_3, output_dim=2).to(device)
elif model_name == 'GraphTransClassifier':
    model = GraphTransClassifier(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, nhead=4, num_encoder_layers=2, output_dim=2).to(device)
elif model_name == 'GraphResidualClassifier':
    model = GraphResidualClassifier(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, hidden_dim_3=hidden_dim_3, output_dim=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Start Training")
 #Eğitim döngüsü
num_epochs = 15
for epoch in range(1, num_epochs + 1):
    loss = train()
    y_true_test, y_pred_test  = test(test_loader)
    accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(y_true_test, y_pred_test)

    if epoch % 3 == 0 or epoch == num_epochs or epoch == 1:
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f} | Test | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("End Training")
# torch.save(model.state_dict(), f'./outputs/{model_name}.pth')
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
result_df.to_csv(f'./outputs/{model_name}_{embedding_type}_results.csv', index=False)
print("Done!")