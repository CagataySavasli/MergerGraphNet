# q_trainer.py - Multi-model trainer for Q_RoBERT, Q_ToBERT, Q_GoBERT

import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from torch_geometric.data import Batch, Data

from lib.database.q_sqlite_dataset import QSQLiteDataset
from lib.model.not_tuned.q_robert import Q_RoBERT
from lib.model.not_tuned.q_tobert import Q_ToBERT
from lib.model.not_tuned.q_gobert import Q_GoBERT

def collate_fn(batch, graph_version=False):
    if graph_version:
        # assert all(isinstance(item[0], Data) for item in batch), "Graph version aktif ama Data objesi gelmedi"
        xs = [item[0] for item in batch]
        xs = Batch.from_data_list(xs)
    else:
        xs = [torch.tensor(item[0], dtype=torch.float32) for item in batch]

    q_x = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    ys = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return xs, q_x, ys

# def collate_fn(batch, graph_version=False):
#     if graph_version:
#         assert all(isinstance(item[0], Data) for item in batch)
#         xs = [item[0] for item in batch]
#         xs = Batch.from_data_list(xs).to(device)
#     else:
#         # önce tüm örnekleri [batch, seq_len, feat] şeklinde bir tensor olarak yığıyoruz
#         xs = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch], dim=0).to(device)
#
#     q_x = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(device)
#     ys = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(device)
#     return xs, q_x, ys

def main():
    model_type = sys.argv[1]  # "q_robert", "q_tobert", "q_gobert"
    GRAPH_VERSION = model_type == "q_gobert"

    # Paths and parameters
    db_path = "./data/database.db"
    q_data_path = "data/auxiliary/wrds/financial_ratios_all_processed.csv"
    table_name = "embeddings"
    date_col = "filing_date"
    val_start = "2019-01-01"
    val_end = "2020-01-01"
    n_days = 365
    seq_len = 12
    batch_size = 8
    epochs = 50
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    train_ds = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=None, end_date=val_start, graph_version=GRAPH_VERSION)
    test_ds  = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=val_start, end_date=val_end, graph_version=GRAPH_VERSION)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda b: collate_fn(b, GRAPH_VERSION))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: collate_fn(b, GRAPH_VERSION))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, q_sample, _ = train_ds[0]
    quant_dim = q_sample.shape[1]

    # Instantiate correct model
    if model_type == "q_robert":
        model = Q_RoBERT(input_dim=768, quant_input_dim=quant_dim, bidirectional=True)
    elif model_type == "q_tobert":
        model = Q_ToBERT(input_dim=768, quant_input_dim=quant_dim)
    elif model_type == "q_gobert":
        model = Q_GoBERT(input_dim=768, quant_input_dim=quant_dim, hidden_dim_1=256, hidden_dim_2=64, lstm_hidden_dim=128)
    else:
        raise ValueError("Unknown model type: " + model_type)

    model.to(device)

    # Compute class weights
    labels = [label for _, _, label in tqdm(train_ds, desc="Extracting labels")]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor(num_neg/num_pos if num_pos > 0 else 1.0).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    best_loss = float('inf')
    history = {k: [] for k in ['epoch', 'train_loss', 'test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1']}
    best_epoch = 0
    print("Training...")
    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for xs, q_x, ys in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training | Best Epoch: {best_epoch}"):
            optimizer.zero_grad()
            logits = model(xs, q_x).squeeze(-1)  # [batch]
            loss = criterion(logits, ys)
            loss.backward()
            # gradient clipping ile aşırı güncellemeleri sınırla
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * ys.size(0)

        avg_train = train_loss / len(train_ds)

        # --- EVAL ---
        model.eval()
        test_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xs, q_x, ys in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} Evaluating"):
                logits = model(xs, q_x).squeeze(-1)
                loss = criterion(logits, ys)
                test_loss += loss.item() * ys.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(ys.cpu().tolist())

        avg_test = test_loss / len(test_ds)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if avg_test < best_loss:
            best_loss = avg_test
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_type}_best_model.pth"))


        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)
        history['test_acc'].append(acc)
        history['test_precision'].append(prec)
        history['test_recall'].append(rec)
        history['test_f1'].append(f1)

        with open(os.path.join(output_dir, f"{model_type}_results.json"), 'w') as f:
            json.dump(history, f, indent=2)

    print("Training complete.")

if __name__ == '__main__':
    main()


# import os
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tqdm import tqdm
# from lib.database.q_sqlite_dataset import QSQLiteDataset
# from lib.model.not_tuned.q_robert import Q_RoBERT
# from lib.model.not_tuned.q_tobert import Q_ToBERT
#
#
#
# def collate_fn(batch):
#     seqs, quants, labels = zip(*batch)
#     xs = [torch.tensor(seq, dtype=torch.float32) for seq in seqs]
#     q_x = torch.tensor(quants, dtype=torch.float32)
#     ys = torch.tensor(labels, dtype=torch.float32)
#     return xs, q_x, ys
#
#
# def main():
#     # Paths and parameters
#     db_path = "./data/database.db"
#     q_data_path = "data/auxiliary/wrds/financial_ratios_all.csv"
#     table_name = "embeddings"
#     date_col = "filing_date"
#     val_start = "2019-01-01"
#     val_end = "2020-01-01"
#     n_days = 365
#     seq_len = 12
#     batch_size = 8
#     epochs = 20
#     output_dir = "outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     # Datasets and loaders
#     print("Loading data...")
#     train_ds = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=None, end_date=val_start)
#     test_ds  = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=val_start, end_date=val_end)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
#
#     device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Determine input dims
#     _, sample_q, _ = train_ds[0]
#     quant_dim = sample_q.shape[1]
#     model = Q_RoBERT(input_dim=768, quant_input_dim=quant_dim, bidirectional=True)
#     model.to(device)
#
#     print("Model Generated")
#
#     # Compute class weight
#     labels = [
#         label
#         for _, _, label in tqdm(
#             train_ds,
#             desc="Extracting labels",
#             unit="item"
#         )
#     ]
#     num_pos = sum(labels)
#     num_neg = len(labels) - num_pos
#     pos_weight = torch.tensor(num_neg/num_pos if num_pos>0 else 1.0).to(device)
#
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     optimizer = optim.Adam(model.parameters(), lr=5e-5)
#
#     best_loss = float('inf')
#     history = {'epoch':[], 'train_loss':[], 'test_loss':[], 'test_acc':[], 'test_precision':[], 'test_recall':[], 'test_f1':[]}
#
#     print("Training...")
#     for epoch in range(1, epochs+1):
#         model.train()
#         train_loss = 0.0
#         for xs, q_x, ys in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training"):
#             xs = [x.to(device) for x in xs]
#             q_x = q_x.to(device)
#             ys = ys.to(device)
#             optimizer.zero_grad()
#             logits = model(xs, q_x)
#             loss = criterion(logits, ys)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * ys.size(0)
#         avg_train = train_loss / len(train_ds)
#
#         # Evaluation
#         model.eval()
#         test_loss = 0.0
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xs, q_x, ys in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} Evaluating"):
#                 xs = [x.to(device) for x in xs]
#                 q_x = q_x.to(device)
#                 ys = ys.to(device)
#                 logits = model(xs, q_x)
#                 loss = criterion(logits, ys)
#                 test_loss += loss.item() * ys.size(0)
#                 preds = (torch.sigmoid(logits) >= 0.5).float()
#                 y_pred.extend(preds.cpu().tolist())
#                 y_true.extend(ys.cpu().tolist())
#         avg_test = test_loss / len(test_ds)
#         acc   = accuracy_score(y_true, y_pred)
#         prec  = precision_score(y_true, y_pred, zero_division=0)
#         rec   = recall_score(y_true, y_pred, zero_division=0)
#         f1    = f1_score(y_true, y_pred, zero_division=0)
#
#         # Save best model
#         if avg_test < best_loss:
#             best_loss = avg_test
#             torch.save(model.state_dict(), os.path.join(output_dir, "q_robert_best_model.pth"))
#
#         # Record history
#         history['epoch'].append(epoch)
#         history['train_loss'].append(avg_train)
#         history['test_loss'].append(avg_test)
#         history['test_acc'].append(acc)
#         history['test_precision'].append(prec)
#         history['test_recall'].append(rec)
#         history['test_f1'].append(f1)
#
#         # Save history
#         with open(os.path.join(output_dir, "q_robert_results.json"), 'w') as f:
#             json.dump(history, f, indent=2)
#
#     print("Training complete.")
#
# if __name__ == '__main__':
#     main()
