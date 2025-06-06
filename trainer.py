import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from lib.model.not_tuned.robert import RoBERT
from lib.model.not_tuned.tobert import ToBERT
from lib.model.not_tuned.gobert import GoBERT
from lib.database.sqlite_dataset import SQLiteDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys

model_type = sys.argv[1]
GRAPH_VERSION = True if model_type == "gobert" else False


def collate_fn(batch):
    # batch: list of tuples (features_list, label)
    if GRAPH_VERSION:
        xs = [example[0] for example in batch]
        xs = Batch.from_data_list(xs)
    else:
        xs = [torch.tensor(example[0], dtype=torch.float32) for example in batch]
    ys = torch.tensor([example[1] for example in batch], dtype=torch.float32)
    return xs, ys

# Dataset and DataLoader
split_date = "2020-01-01"
train_ds = SQLiteDataset(
    db_path="./data/database.db",
    table_name="embeddings",
    date_column="filing_date",
    end_date=split_date,
    graph_version=GRAPH_VERSION,
    conn_sentence=2
)
test_ds = SQLiteDataset(
    db_path="./data/database.db",
    table_name="embeddings",
    date_column="filing_date",
    start_date=split_date,
    graph_version=GRAPH_VERSION,
    conn_sentence=2
)
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_fn
)


# --- Training Setup ---
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_type == "robert":
    model = RoBERT(input_dim=768, hidden_dim=128, num_layers=1, bidirectional=False).to(device)
elif model_type == "tobert":
    model = ToBERT(input_dim=768, num_heads=8, num_layers=2, dim_feedforward=2048).to(device)
else:
    model = GoBERT(input_dim=768, hidden_dim_1=256, hidden_dim_2=64, output_dim=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs = 50
print("Starting training...")
for epoch in range(1, epochs+1):
    print("Starting epoch:", epoch)
    model.train()
    total_loss = 0.0
    train_bar = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc="Training",
        ncols=80
    )

    for batch_idx, (xs, ys) in train_bar:
        # convert lists to tensors and move to device
        if not GRAPH_VERSION:
            xs = [x.to(device) for x in xs]
        ys = ys.to(device)

        optimizer.zero_grad()
        logits = model(xs)
        # print(ys)
        # print("logits:", logits)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ys.size(0)

    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f}")

    print("Evaluating model...")
    model.eval()
    correct, total = 0, 0

    y_reals = []
    y_preds = []
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
            preds = (torch.sigmoid(logits) >= 0.5).float()

            y_preds.extend(preds.cpu().numpy())
            y_reals.extend(ys.cpu().numpy())

    acc = accuracy_score(y_reals, y_preds)
    prec = precision_score(y_reals, y_preds)
    rec = recall_score(y_reals, y_preds)
    f1 = f1_score(y_reals, y_preds)

    tn, fp, fn, tp = confusion_matrix(y_reals, y_preds).ravel()
    print(f"Epoch {epoch}/{epochs} - Test | Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
