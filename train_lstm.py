"""Train the LSTM once and save the results; eda_cleaning.ipynb only loads them.

    python train_lstm.py            # train seed 4266 -> models/lstm.pt, models/lstm_history.csv
    python train_lstm.py --sweep    # also train other seeds -> models/lstm_seed_runs.csv
Importing this module only loads the data and defines helpers; it never trains - Have to run this .py file directly to train.
"""
import argparse
import copy
from pathlib import Path

import numpy as np  # keep before torch (OMP libiomp5md clash on Windows/anaconda)
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lstm.pt"
HISTORY_PATH = MODEL_DIR / "lstm_history.csv"
SEED_RUNS_PATH = MODEL_DIR / "lstm_seed_runs.csv"

CLASSES = ["non_vpn", "vpn", "tor"]
BLOCKS = ["bytes_out", "bytes_in", "packets_out", "packets_in",
          "gap_mean", "gap_std", "size_mean", "size_std"]
N_BUCKETS = 28   # NUM_BUCKETS used with T_MAX = 5s
SEEDS = [4266, 1, 2, 3, 4]

data = np.load("data/final_data.npz")

def to_seq(X):
    # (N, 224) block-major -> (N, 28 time steps, 8 channels)
    X = X.reshape(len(X), len(BLOCKS), N_BUCKETS).transpose(0, 2, 1)
    return np.ascontiguousarray(X, dtype=np.float32)

label_id = {c: i for i, c in enumerate(CLASSES)}
X_train, X_validation, X_test = (to_seq(data[k]) for k in ("X_train", "X_val", "X_test"))
y_train, y_validation, y_test = (np.array([label_id[v] for v in data[k]]) for k in ("y_train", "y_val", "y_test"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RNNClassifier(nn.Module):
    """LSTM (default) or GRU over the 28 time steps, pooled over time, then a linear head.

    pooling: "mean" (default), "max", or "masked_mean", which averages only the non-empty steps and
    expects a 0/1 non-empty flag as the LAST input channel. The defaults reproduce the saved lstm.pt.
    """
    def __init__(self, cell="lstm", n_in=len(BLOCKS), hidden=32, layers=2, n_classes=len(CLASSES), dropout=0.2,
                 bidirectional=False, pooling="mean"):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.pooling = pooling
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)                   # (B, T, hidden)
        if self.pooling == "max":
            pooled = out.max(dim=1).values
        elif self.pooling == "masked_mean":
            keep = x[..., -1:]
            pooled = (out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)           # mean pool over the 28 time steps
        return self.head(pooled)

def predict(model, X):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=1024)
    with torch.no_grad():
        logits = torch.cat([model(xb.to(DEVICE)).cpu() for (xb,) in loader])
    return logits.argmax(1).numpy()

def train_torch(make_model, X_tr, y_tr, X_va=None, y_va=None, seed=4266, epochs=60, patience=8, lr=1e-3,
                batch_size=256, weight_decay=0.0, class_weight_power=1.0, restore_best=True):
    """Train any torch model on (X_tr, y_tr); keep the epoch with the best val macro-F1.

    Without (X_va, y_va) there is no early stopping and the last epoch is kept. restore_best=False also
    keeps the last epoch (used for fixed-epoch training and for recording full learning curves).
    """
    torch.manual_seed(seed)
    model = make_model().to(DEVICE)

    counts = np.bincount(y_tr, minlength=len(CLASSES))
    class_w = torch.tensor((counts.sum() / (len(CLASSES) * counts)) ** class_weight_power,
                           dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_f1, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)

        row = {"epoch": epoch, "train_loss": total / len(y_tr)}
        if X_va is None:
            history.append(row)
            continue

        pred = predict(model, X_va)
        val_f1 = f1_score(y_va, pred, average="macro")
        per_class = f1_score(y_va, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
        history.append({**row, "val_macro_f1": val_f1, **{f"f1_{c}": v for c, v in zip(CLASSES, per_class)}})
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)   # restore the epoch with the best val macro-F1
    return model, pd.DataFrame(history)

def train_lstm(seed=4266, **kwargs):
    return train_torch(lambda: RNNClassifier("lstm"), X_train, y_train, X_validation, y_validation, seed, **kwargs)

def score_row(y_true, pred, **extra):
    """macro-F1 and per-class F1 for one set of predictions; `extra` columns (model, split, seed...) come first."""
    labels = range(len(CLASSES))
    per_class = f1_score(y_true, pred, average=None, labels=labels, zero_division=0)
    return {**extra, "macro_f1": f1_score(y_true, pred, average="macro", labels=labels, zero_division=0),
            **{f"f1_{c}": v for c, v in zip(CLASSES, per_class)}}

def load_model(path=MODEL_PATH):
    """Rebuild the LSTM and load saved weights, ready for inference."""
    model = RNNClassifier("lstm").to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    state = {k.replace("lstm.", "rnn.", 1): v for k, v in state.items()}   # checkpoints saved before the rename
    model.load_state_dict(state)
    return model.eval()

def load_history(path=HISTORY_PATH):
    return pd.read_csv(path)

def seed_sweep(seeds=SEEDS, trained=None):
    """Score every seed; `trained` maps seed -> already-trained model so it isn't trained twice."""
    trained = trained or {}
    rows = []
    for seed in seeds:
        model = trained[seed] if seed in trained else train_lstm(seed=seed)[0]
        for split, X, y in (("val", X_validation, y_validation), ("test", X_test, y_test)):
            rows.append(score_row(y, predict(model, X), seed=seed, split=split))
        print(f"seed {seed} done")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--sweep", action="store_true", help="also train every seed in SEEDS and save the scores")
    args = parser.parse_args()

    print(X_train.shape, X_validation.shape, X_test.shape)
    print(pd.DataFrame({"train": np.bincount(y_train), "val": np.bincount(y_validation),
                        "test": np.bincount(y_test)}, index=CLASSES))

    MODEL_DIR.mkdir(exist_ok=True)
    model, history = train_lstm(seed=args.seed)
    torch.save(model.state_dict(), MODEL_PATH)
    history.to_csv(HISTORY_PATH, index=False)
    best_epoch = int(history["val_macro_f1"].idxmax()) + 1
    print(f"device: {DEVICE} | epochs run: {len(history)} | best epoch: {best_epoch} "
          f"| best val macro-F1: {history['val_macro_f1'].max():.3f}")
    print(f"saved {MODEL_PATH} and {HISTORY_PATH}")

    if args.sweep:
        seed_sweep(trained={args.seed: model}).to_csv(SEED_RUNS_PATH, index=False)
        print(f"saved {SEED_RUNS_PATH}")
