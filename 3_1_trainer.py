import os
import json
import sys
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from lib.model.not_tuned.robert import RoBERT
from lib.model.not_tuned.tobert import ToBERT
from lib.model.not_tuned.gobert import GoBERT
from lib.database.sqlite_dataset import SQLiteDataset

# Class distribution (provided)
num_1_label = 1339
num_0_label = 6539

model_type = sys.argv[1]
GRAPH_VERSION = True if model_type == "gobert" else False
torch.cuda.empty_cache()

def collate_fn(batch, graph_version=False):
    if graph_version:
        xs = [example[0] for example in batch]
        xs = Batch.from_data_list(xs)
    else:
        xs = [torch.tensor(example[0], dtype=torch.float32) for example in batch]
    ys = torch.tensor([example[1] for example in batch], dtype=torch.float32)
    return xs, ys


def main():
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Dataset and DataLoader
    val_str_date = "2019-01-01"
    val_end_date = "2020-01-01"
    train_ds = SQLiteDataset(
        db_path="./data/database.db",
        table_name="embeddings",
        date_column="filing_date",
        end_date=val_str_date,
        graph_version=GRAPH_VERSION,
        conn_sentence=2
    )
    test_ds = SQLiteDataset(
        db_path="./data/database.db",
        table_name="embeddings",
        date_column="filing_date",
        start_date=val_str_date,
        end_date=val_end_date,
        graph_version=GRAPH_VERSION,
        conn_sentence=2
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # --- Training Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "robert":
        model = RoBERT(input_dim=768, hidden_dim=128, num_layers=1, bidirectional=False)
    elif model_type == "tobert":
        model = ToBERT(input_dim=768, num_heads=8, num_layers=2, dim_feedforward=2048)
    else:
        model = GoBERT(input_dim=768, hidden_dim_1=256, hidden_dim_2=64, output_dim=1)

    # Loss with pos_weight to penalize minority class less
    model.to(device)
    pos_weight = torch.tensor(num_0_label / num_1_label).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    epochs = 300  # Increased from 50 to 300
    print("Starting training...")

    # Best model tracking
    best_test_loss = float('inf')

    all_results = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'tp': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'is_best': []
    }

    output_path = f"outputs/{model_type}_results.json"
    best_model_path = f"outputs/{model_type}_best_model.pth"

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} - Training...")
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc="Training",
            ncols=80
        )

        for batch_idx, (xs, ys) in train_bar:
            if not GRAPH_VERSION:
                xs = [x.to(device) for x in xs]
            ys = ys.to(device)

            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * ys.size(0)

        avg_train_loss = total_train_loss / len(train_ds)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Evaluate
        print(f"Epoch {epoch}/{epochs} - Evaluating...")
        model.eval()
        total_test_loss = 0.0
        y_reals, y_preds = [], []

        with torch.no_grad():
            test_bar = tqdm(
                enumerate(test_loader, start=1),
                total=len(test_loader),
                desc="Evaluating",
                ncols=80
            )
            for batch_idx, (xs, ys) in test_bar:
                if not GRAPH_VERSION:
                    xs = [x.to(device) for x in xs]
                ys = ys.to(device)

                logits = model(xs)
                loss = criterion(logits, ys)
                total_test_loss += loss.item() * ys.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                y_preds.extend(preds.cpu().numpy())
                y_reals.extend(ys.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_ds)
        acc = accuracy_score(y_reals, y_preds)
        prec = precision_score(y_reals, y_preds, zero_division=True)
        rec = recall_score(y_reals, y_preds, zero_division=True)
        f1 = f1_score(y_reals, y_preds, zero_division=True)
        tn, fp, fn, tp = confusion_matrix(y_reals, y_preds).ravel()

        print(f"Epoch {epoch}/{epochs} - Test Loss: {avg_test_loss:.4f} | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        # Determine if best
        is_best = False
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch} with Test Loss: {avg_test_loss:.4f}")
            is_best = True

        # Record results
        all_results['epoch'].append(epoch)
        all_results['train_loss'].append(avg_train_loss)
        all_results['test_loss'].append(avg_test_loss)
        all_results['test_acc'].append(acc)
        all_results['test_precision'].append(prec)
        all_results['test_recall'].append(rec)
        all_results['test_f1'].append(f1)
        all_results['tp'].append(tp)
        all_results['tn'].append(tn)
        all_results['fp'].append(fp)
        all_results['fn'].append(fn)
        all_results['is_best'].append(is_best)

        # Ensure all types are JSON serializable
        for key, values in all_results.items():
            all_results[key] = [v.item() if hasattr(v, 'item') else v for v in values]

        # Save intermediate results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Results up to epoch {epoch} saved to {output_path}")

    print("Training complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()