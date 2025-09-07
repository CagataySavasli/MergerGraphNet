# q_tester.py - Multi-model tester for Q_RoBERT, Q_ToBERT, Q_GoBERT

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from lib.database.q_sqlite_dataset import QSQLiteDataset
from lib.model.not_tuned.q_robert import Q_RoBERT
from lib.model.not_tuned.q_tobert import Q_ToBERT
from lib.model.not_tuned.q_gobert import Q_GoBERT

def collate_fn(batch, graph_version=False):
    if graph_version:
        xs = [example[0] for example in batch]
        xs = Batch.from_data_list(xs)
    else:
        xs = [torch.tensor(example[0], dtype=torch.float32) for example in batch]
    q_x = torch.tensor([example[1] for example in batch], dtype=torch.float32)
    ys = torch.tensor([example[2] for example in batch], dtype=torch.float32)
    return xs, q_x, ys

def main():
    if len(sys.argv) < 2:
        print("Usage: python q_tester.py <model_type>")
        sys.exit(1)

    model_type = sys.argv[1]
    GRAPH_VERSION = model_type == "q_gobert"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data and config
    db_path = "./data/database.db"
    q_data_path = "data/auxiliary/wrds/financial_ratios_all_processed.csv"
    table_name = "embeddings"
    date_col = "filing_date"
    val_end = "2020-01-01"
    n_days = 365
    seq_len = 12
    batch_size = 8

    test_ds = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=val_end, end_date=None, graph_version=GRAPH_VERSION)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: collate_fn(b, GRAPH_VERSION))

    # Model selection
    _, q_sample, _ = test_ds[0]
    quant_dim = q_sample.shape[1]
    if model_type == "q_robert":
        model = Q_RoBERT(input_dim=768, quant_input_dim=quant_dim, bidirectional=True)
    elif model_type == "q_tobert":
        model = Q_ToBERT(input_dim=768, quant_input_dim=quant_dim)
    elif model_type == "q_gobert":
        model = Q_GoBERT(input_dim=768, quant_input_dim=quant_dim, hidden_dim_1=256, hidden_dim_2=64, lstm_hidden_dim=128)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load weights
    model_path = os.path.join("outputs", f"{model_type}_best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for xs, q_x, ys in tqdm(test_loader, desc="Testing"):
            xs = [x.to(device) for x in xs] if not GRAPH_VERSION else xs.to(device)
            q_x = q_x.to(device)
            ys = ys.to(device)
            logits = model(xs, q_x)
            loss = criterion(logits, ys)
            losses.append(loss.item() * ys.size(0))
            preds = (torch.sigmoid(logits) >= 0.5).float()
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(ys.cpu().tolist())

    # Metrics
    avg_loss = sum(losses) / len(test_ds)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        'test_loss': avg_loss,
        'test_acc': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

    print(json.dumps(results, indent=2))
    out_path = os.path.join("outputs", f"{model_type}_test_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()



# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from tqdm import tqdm
# from lib.database.q_sqlite_dataset import QSQLiteDataset
# from lib.model.not_tuned.q_robert import Q_RoBERT
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
#     val_end = "2020-01-01"
#     n_days = 365
#     seq_len = 12
#     batch_size = 8
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Test dataset and loader
#     test_ds = QSQLiteDataset(db_path, table_name, date_col, q_data_path, n_days, seq_len, start_date=val_end, end_date=None)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
#
#     # Load model
#     _, sample_q, _ = test_ds[0]
#     quant_dim = sample_q.shape[1]
#     model = Q_RoBERT(input_dim=768, quant_input_dim=quant_dim, bidirectional=True)
#     model.load_state_dict(torch.load(os.path.join("outputs", "q_robert_best_model.pth"), map_location=device))
#     model.to(device)
#     model.eval()
#
#     criterion = nn.BCEWithLogitsLoss()
#     losses = []
#     y_true, y_pred = [], []
#
#     with torch.no_grad():
#         for xs, q_x, ys in tqdm(test_loader, desc="Testing"):
#             xs = [x.to(device) for x in xs]
#             q_x = q_x.to(device)
#             ys = ys.to(device)
#             logits = model(xs, q_x)
#             loss = criterion(logits, ys)
#             losses.append(loss.item() * ys.size(0))
#             preds = (torch.sigmoid(logits) >= 0.5).float()
#             y_pred.extend(preds.cpu().tolist())
#             y_true.extend(ys.cpu().tolist())
#
#     avg_loss = sum(losses) / len(test_ds)
#     acc   = accuracy_score(y_true, y_pred)
#     prec  = precision_score(y_true, y_pred, zero_division=0)
#     rec   = recall_score(y_true, y_pred, zero_division=0)
#     f1    = f1_score(y_true, y_pred, zero_division=0)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#
#     results = {
#         'test_loss': avg_loss,
#         'test_acc': acc,
#         'test_precision': prec,
#         'test_recall': rec,
#         'test_f1': f1,
#         'tp': int(tp),
#         'tn': int(tn),
#         'fp': int(fp),
#         'fn': int(fn)
#     }
#
#     print(results)
#     with open(os.path.join("outputs", "q_robert_test_results.json"), 'w') as f:
#         json.dump(results, f, indent=2)
#
# if __name__ == '__main__':
#     main()
#
# # {
# #   "test_loss": 0.6591820419451712,
# #   "test_acc": 0.6097308488612836,
# #   "test_precision": 0.16714697406340057,
# #   "test_recall": 0.3972602739726027,
# #   "test_f1": 0.23529411764705882,
# #   "tp": 58,
# #   "tn": 531,
# #   "fp": 289,
# #   "fn": 88
# # }