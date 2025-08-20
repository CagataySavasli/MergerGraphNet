import os
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from lib.model.not_tuned.robert import RoBERT
from lib.model.not_tuned.tobert import ToBERT
from lib.model.not_tuned.gobert import GoBERT
from lib.database.sqlite_dataset import SQLiteDataset

# Class distribution if needed for pos_weight, but only eval so not used here
# num_1_label = 1339
# num_0_label = 6539

def collate_fn(batch, graph_version=False):
    if graph_version:
        xs = [example[0] for example in batch]
        xs = Batch.from_data_list(xs)
    else:
        xs = [torch.tensor(example[0], dtype=torch.float32) for example in batch]
    ys = torch.tensor([example[1] for example in batch], dtype=torch.float32)
    return xs, ys


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_trainer.py <model_type>")
        sys.exit(1)

    model_type = sys.argv[1]
    GRAPH_VERSION = True if model_type == "gobert" else False
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define date for test split start (data after validation end date)
    val_end_date = "2020-01-01"

    # Prepare test dataset and loader
    test_ds = SQLiteDataset(
        db_path="./data/database.db",
        table_name="embeddings",
        date_column="filing_date",
        start_date=val_end_date,
        graph_version=GRAPH_VERSION,
        conn_sentence=2
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, GRAPH_VERSION)
    )

    # Instantiate model
    if model_type == "robert":
        model = RoBERT(input_dim=768, hidden_dim=128, num_layers=1, bidirectional=False)
    elif model_type == "tobert":
        model = ToBERT(input_dim=768, num_heads=8, num_layers=2, dim_feedforward=2048)
    else:
        model = GoBERT(input_dim=768, hidden_dim_1=256, hidden_dim_2=64, output_dim=1)
    model = model.to(device)

    # Load trained weights
    model_path = os.path.join("outputs_den", f"{model_type}_best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation
    y_true, y_pred = [], []
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xs, ys in tqdm(test_loader, desc="Testing", ncols=80):
            if GRAPH_VERSION:
                xs = xs.to(device)
            else:
                xs = [x.to(device) for x in xs]
            ys = ys.to(device)

            logits = model(xs)
            loss = criterion(logits, ys)
            losses.append(loss.item() * ys.size(0))

            preds = (torch.sigmoid(logits) >= 0.5).float()
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(ys.cpu().tolist())

    # Compute metrics
    avg_loss = sum(losses) / len(test_ds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Prepare result dict
    results = {
        'test_loss': avg_loss,
        'test_acc': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    print(results)
    # Save to JSON
    # os.makedirs("outputs", exist_ok=True)
    # out_file = os.path.join("outputs", f"{model_type}_test_results.json")
    # with open(out_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2)
    # print(f"Test results saved to {out_file}")


if __name__ == "__main__":
    main()
