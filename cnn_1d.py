"""1D CNN for classifying flows as non_vpn / vpn / tor.

Uses the bucketed features in data/final_data.npz (8 channels x 28 time buckets per flow).
The npz is already log1p + standard scaled in eda_cleaning.ipynb, so we don't scale again here.

Usage:
    python cnn_1d.py
    python cnn_1d.py --channels size timing          # channel-subset experiment
    python cnn_1d.py --no-class-weights              # class-weighting experiment
    python cnn_1d.py --cv-folds 5                    # grouped CV + tor threshold tuning
    python cnn_1d.py --tor-threshold 0.6             # use a threshold picked from CV
    python cnn_1d.py --evaluate-test                 # final run only

Inference in a notebook:
    from cnn_1d import load_model, predict
    model, checkpoint = load_model("outputs/cnn_1d/<run>/best_model.pt")
    labels, probabilities = predict(model, checkpoint, X_test)   # X_test straight from final_data.npz
"""

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["non_vpn", "vpn", "tor"]
NUM_BUCKETS = 28

# order of the 28-column blocks in feature_cols
CHANNEL_NAMES = [
    "bytes_out", "bytes_in",
    "packets_out", "packets_in",
    "gap_mean", "gap_std",
    "size_mean", "size_std",
]

# channel groups for the size vs timing experiment
CHANNEL_GROUPS = {
    "volume": ["bytes_out", "bytes_in", "packets_out", "packets_in"],
    "timing": ["gap_mean", "gap_std"],
    "size": ["size_mean", "size_std"],
}

# packet-count bands for the results breakdown (packets inside the 5s window)
# most flows only have 1-2 packets, so we also report results per band
PACKET_COUNT_BANDS = [(1, 2), (3, 10), (11, None)]


def parse_arguments():
    """
    CLI comments so the script can run on cloud
    and each experiment is a single command with no code edits.

    Data and output:
        --data-path         npz produced by eda_cleaning.ipynb (default: data/final_data.npz)
        --output-dir        parent folder; each run gets its own timestamped subfolder
        --channels          input channels: 'all', group names (volume, timing, size)
                            or individual channel names, e.g. --channels size gap_mean

    Training:
        --epochs            maximum number of epochs
        --patience          early-stopping patience on validation macro-F1
        --batch-size        flows per batch
        --learning-rate     AdamW learning rate
        --weight-decay      L2 penalty applied by AdamW (0 disables it)
        --dropout           dropout rate after each conv block and in the classifier
        --no-class-weights  use an unweighted loss instead of balanced class weights
        --seed              random seed for weight init, shuffling and dropout

    Model:
        --conv-channels     output channels of each conv block, e.g. --conv-channels 32 64
        --kernel-size       convolution width, in time buckets

    Evaluation:
        --evaluate-test     also score the test set; use for the final run only
        --cv-folds          k > 0 runs k-fold cross-validation grouped by stream_id over
                            train+val instead of the fixed split; the test set is not used
        --tor-threshold     predict tor whenever P(tor) >= this value, otherwise the
                            likelier of non_vpn / vpn; pick it from a --cv-folds run
        --scaler-path       scaler saved with the npz; used only to recover packet counts
                            for the per-flow-size breakdown (skipped if the file is missing)

    Returns:
        argparse.Namespace with one attribute per option (dashes become underscores,
        e.g. arguments.learning_rate).
    """
    parser = argparse.ArgumentParser(description="Train a 1D CNN on bucketed flow features.")
    parser.add_argument("--data-path", type=Path, default=Path("data/final_data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn_1d"))
    parser.add_argument("--channels", nargs="+", default=["all"],
                        help="'all', group names (volume, timing, size) or channel names")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="stop after this many epochs without val macro-F1 improvement")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty (AdamW)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--conv-channels", nargs="+", type=int, default=[32, 64],
                        help="output channels of each conv block")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also report on the test set (use for the final run only)")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="k-fold cross-validation grouped by stream over train+val (0 = fixed split)")
    parser.add_argument("--tor-threshold", type=float, default=None,
                        help="predict tor when P(tor) >= this value (default: plain argmax)")
    parser.add_argument("--scaler-path", type=Path, default=Path("data/scaler.joblib"),
                        help="scaler used for the npz, to recover packet counts for the size breakdown")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        '''
        MPS is some mac thing, for small level local dev
        '''
        return torch.device("mps")
    return torch.device("cpu")


def resolve_channels(requested):
    """Turns the --channels input (all / group names / channel names) into a list of channels."""
    if "all" in requested:
        return list(CHANNEL_NAMES)
    selected = []
    for name in requested:
        if name in CHANNEL_GROUPS:
            selected.extend(CHANNEL_GROUPS[name])
        elif name in CHANNEL_NAMES:
            selected.append(name)
        else:
            raise ValueError(f"unknown channel or group: {name}")
    # keep the original channel order and remove duplicates
    return [channel for channel in CHANNEL_NAMES if channel in selected]


def prepare_features(flat_features, channels):
    """Reshapes (flows, 224) into (flows, channels, 28) for Conv1d, keeping only the chosen channels."""
    channel_indices = [CHANNEL_NAMES.index(channel) for channel in channels]
    grid = np.asarray(flat_features, dtype=np.float32).reshape(-1, len(CHANNEL_NAMES), NUM_BUCKETS)
    return np.ascontiguousarray(grid[:, channel_indices, :])


def load_split_data(data_path, channels, scaler_path=None):
    """Loads the npz.

    Returns:
        splits: dict of split -> (features, labels, stream_ids)
        augmented_mask: True for the augmented (synthetic) rows in train
        packet_counts: dict of split -> packets per flow, or None if the scaler file is missing
    """
    archive = np.load(data_path)

    expected_columns = [f"{channel}_{bucket}" for channel in CHANNEL_NAMES for bucket in range(NUM_BUCKETS)]
    if list(archive["feature_cols"]) != expected_columns:
        raise ValueError("feature_cols order differs from the expected 8 x 28 layout")

    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    splits = {}
    for split in ("train", "val", "test"):
        features = prepare_features(archive[f"X_{split}"], channels)
        labels = np.array([label_to_index[label] for label in archive[f"y_{split}"]], dtype=np.int64)
        splits[split] = (features, labels, archive[f"groups_{split}"].astype(str))

    packet_counts = None
    if scaler_path is not None and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        packet_counts = {split: count_packets_in_window(archive[f"X_{split}"], scaler)
                         for split in ("train", "val", "test")}
    return splits, find_augmented_rows(archive, splits["train"][1]), packet_counts


def count_packets_in_window(flat_features, scaler):
    """Undoes the scaling to get the raw packet count of each flow (packets_out + packets_in)."""
    raw_features = np.expm1(scaler.inverse_transform(flat_features))
    packet_blocks = [CHANNEL_NAMES.index("packets_out"), CHANNEL_NAMES.index("packets_in")]
    packet_columns = [block * NUM_BUCKETS + bucket for block in packet_blocks for bucket in range(NUM_BUCKETS)]
    return np.round(raw_features[:, packet_columns].sum(axis=1)).astype(int)


def find_augmented_rows(archive, train_labels):
    """Finds which train rows are augmented.

    The npz doesn't store this, but eda_cleaning.ipynb appends the augmented tor rows
    at the end of train, so we count the tor rows at the end.
    If the npz ever gets an "is_augmented_train" array, that is used instead.
    """
    if "is_augmented_train" in archive.files:
        return archive["is_augmented_train"].astype(bool)
    tor_index = CLASS_NAMES.index("tor")
    trailing_tor_rows = 0
    for label in train_labels[::-1]:
        if label != tor_index:
            break
        trailing_tor_rows += 1
    augmented_mask = np.zeros(len(train_labels), dtype=bool)
    augmented_mask[len(train_labels) - trailing_tor_rows:] = True
    return augmented_mask


def make_loader(features, labels, batch_size, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_class_weights(labels):
    """Same formula as sklearn's class_weight='balanced'."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = len(labels) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


class FlowCNN(nn.Module):
    """Conv blocks over the time buckets, then a small fully connected classifier.

    Each block halves the number of time steps (28 -> 14 -> 7). We flatten at the end instead
    of global pooling, so the model still knows roughly when in the flow something happened.
    """

    def __init__(self, num_input_channels, num_classes, conv_channels, kernel_size, dropout):
        super().__init__()
        blocks = []
        in_channels = num_input_channels
        time_steps = NUM_BUCKETS
        for out_channels in conv_channels:
            blocks += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
            time_steps //= 2
        self.feature_extractor = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * time_steps, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, flow_buckets):
        return self.classifier(self.feature_extractor(flow_buckets))


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_features, batch_labels in loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(batch_features), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_function, device):
    """Returns (average loss, true labels, predicted labels)."""
    model.eval()
    total_loss = 0.0
    true_labels, predicted_labels = [], []
    for batch_features, batch_labels in loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        logits = model(batch_features)
        total_loss += loss_function(logits, batch_labels).item() * len(batch_labels)
        true_labels.append(batch_labels.cpu().numpy())
        predicted_labels.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(true_labels), np.concatenate(predicted_labels)


def macro_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, labels=range(len(CLASS_NAMES)),
                    average="macro", zero_division=0)


def apply_tor_threshold(probabilities, tor_threshold=None):
    """Turns probabilities into predicted classes.

    No threshold = normal argmax. With a threshold, a flow is only predicted as tor if
    P(tor) >= threshold, otherwise it gets whichever of non_vpn / vpn is more likely.
    """
    if tor_threshold is None:
        return probabilities.argmax(axis=1)
    tor_index = CLASS_NAMES.index("tor")
    other_class_probabilities = probabilities.copy()
    other_class_probabilities[:, tor_index] = -1.0
    predictions = other_class_probabilities.argmax(axis=1)
    predictions[probabilities[:, tor_index] >= tor_threshold] = tor_index
    return predictions


def tune_tor_threshold(true_labels, probabilities):
    """Tries thresholds from 0.05 to 0.95 and returns the one with the best macro-F1."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [macro_f1(true_labels, apply_tor_threshold(probabilities, candidate)) for candidate in candidates]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


def describe_results(split_name, true_labels, predicted_labels):
    """Classification report and confusion matrix as a string."""
    report = classification_report(true_labels, predicted_labels, labels=range(len(CLASS_NAMES)),
                                   target_names=CLASS_NAMES, digits=3, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(true_labels, predicted_labels, labels=range(len(CLASS_NAMES))),
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    )
    return (f"=== {split_name} | macro-F1: {macro_f1(true_labels, predicted_labels):.3f} ===\n"
            f"{report}\nconfusion matrix (rows = true, cols = predicted)\n{matrix}\n")


def describe_by_packet_count(split_name, true_labels, predicted_labels, packet_counts):
    """Results split by flow size (packets in the window), to see how tiny flows compare to bigger ones."""
    rows = []
    for lowest, highest in PACKET_COUNT_BANDS:
        if highest is None:
            in_band = packet_counts >= lowest
            band_name = f"{lowest}+"
        else:
            in_band = (packet_counts >= lowest) & (packet_counts <= highest)
            band_name = f"{lowest}-{highest}"
        band_true = true_labels[in_band]
        band_predicted = predicted_labels[in_band]

        row = {
            "packets": band_name,
            "flows": int(in_band.sum()),
            "share": in_band.mean(),
            "tor_flows": int((band_true == CLASS_NAMES.index("tor")).sum()),
            "macro_f1": macro_f1(band_true, band_predicted) if in_band.any() else float("nan"),
        }
        per_class_f1 = f1_score(band_true, band_predicted, labels=range(len(CLASS_NAMES)),
                                average=None, zero_division=0)
        for name, score in zip(CLASS_NAMES, per_class_f1):
            row[f"f1_{name}"] = score
        rows.append(row)
    table = pd.DataFrame(rows)
    return table, f"--- {split_name}: by flow size (packets in window) ---\n{table.round(3).to_string(index=False)}\n"


def save_checkpoint(model, model_config, channels, checkpoint_path, extra_info=None):
    """Save weights together with everything needed to rebuild the model. (for inference)"""
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model_config,
        "channels": channels,
        "class_names": CLASS_NAMES,
        "num_buckets": NUM_BUCKETS,
        **(extra_info or {}),
    }, checkpoint_path)


def load_model(checkpoint_path, device="cpu"):
    """Loads a saved checkpoint and rebuilds the model. Returns (model, checkpoint)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FlowCNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def compute_probabilities(model, features, batch_size=1024):
    """Softmax probabilities for each flow, done in batches."""
    model.eval()
    device = next(model.parameters()).device
    feature_tensor = torch.from_numpy(features)
    probability_batches = []
    for start in range(0, len(feature_tensor), batch_size):
        logits = model(feature_tensor[start:start + batch_size].to(device))
        probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probability_batches)


def predict(model, checkpoint, flat_features, tor_threshold=None, batch_size=1024):
    """Predicts classes for X_train / X_val / X_test taken straight from final_data.npz.

    If tor_threshold is not given, uses the one saved in the checkpoint (or plain argmax if none).
    Returns (predicted class names, probabilities). Probability columns are in checkpoint["class_names"] order.
    """
    if tor_threshold is None:
        tor_threshold = checkpoint.get("tor_threshold")
    probabilities = compute_probabilities(model, prepare_features(flat_features, checkpoint["channels"]), batch_size)
    predicted_names = np.array(checkpoint["class_names"])[apply_tor_threshold(probabilities, tor_threshold)]
    return predicted_names, probabilities


def plot_history(history, output_path):
    import matplotlib

    matplotlib.use("Agg")  # only saving to file. imported here so it doesn't break inline plots in notebooks
    import matplotlib.pyplot as plt

    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="val")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)
    f1_axis.plot(history["epoch"], history["val_macro_f1"], label="val macro-F1")
    f1_axis.set_xlabel("epoch")
    f1_axis.set_ylabel("macro-F1")
    f1_axis.legend()
    f1_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_model_config(arguments, channels):
    return {
        "num_input_channels": len(channels),
        "num_classes": len(CLASS_NAMES),
        "conv_channels": arguments.conv_channels,
        "kernel_size": arguments.kernel_size,
        "dropout": arguments.dropout,
    }


def train_model(model_config, channels, train_set, val_set, arguments, device, checkpoint_path, log_prefix=""):
    """Trains one model, early stopping on val macro-F1, and returns the best epoch's model.

    train_set and val_set are (features, labels). Returns (model, history, best_epoch).
    """
    set_seed(arguments.seed)
    train_features, train_labels = train_set
    train_loader = make_loader(train_features, train_labels, arguments.batch_size, True, arguments.seed)
    val_loader = make_loader(*val_set, arguments.batch_size, False, arguments.seed)

    model = FlowCNN(**model_config).to(device)
    class_weights = None if arguments.no_class_weights else compute_class_weights(train_labels).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate,
                                  weight_decay=arguments.weight_decay)

    history_rows = []
    best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0
    for epoch in range(1, arguments.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_true, val_predicted = evaluate(model, val_loader, loss_function, device)
        val_f1 = macro_f1(val_true, val_predicted)
        history_rows.append({"epoch": epoch, "train_loss": train_loss,
                             "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"{log_prefix}epoch {epoch:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_improvement = val_f1, epoch, 0
            save_checkpoint(model, model_config, channels, checkpoint_path,
                            {"best_epoch": best_epoch, "val_macro_f1": best_f1, "seed": arguments.seed,
                             "tor_threshold": arguments.tor_threshold})
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= arguments.patience:
                print(f"{log_prefix}early stopping: no improvement for {arguments.patience} epochs")
                break

    best_model, _ = load_model(checkpoint_path, device)
    return best_model, pd.DataFrame(history_rows), best_epoch


def run_fixed_split(arguments, splits, packet_counts, channels, model_config, device, run_dir):
    """Normal run: train on train, early stop on val, and score test if --evaluate-test is set."""
    train_features, train_labels, _ = splits["train"]
    val_features, val_labels, _ = splits["val"]

    model, history, best_epoch = train_model(
        model_config, channels, (train_features, train_labels), (val_features, val_labels),
        arguments, device, run_dir / "best_model.pt")
    history.to_csv(run_dir / "history.csv", index=False)
    plot_history(history, run_dir / "training_curves.png")

    threshold_note = "argmax" if arguments.tor_threshold is None else f"tor threshold {arguments.tor_threshold}"
    val_predicted = apply_tor_threshold(compute_probabilities(model, val_features), arguments.tor_threshold)
    summary_text = (f"best epoch: {best_epoch} | decision rule: {threshold_note}\n\n"
                    + describe_results("validation", val_labels, val_predicted))
    metrics = {"best_epoch": best_epoch, "val_macro_f1": macro_f1(val_labels, val_predicted)}
    if packet_counts is not None:
        size_table, size_text = describe_by_packet_count("validation", val_labels, val_predicted, packet_counts["val"])
        summary_text += "\n" + size_text
        metrics["val_macro_f1_by_packets"] = dict(zip(size_table["packets"], size_table["macro_f1"]))

    if arguments.evaluate_test:
        test_features, test_labels, _ = splits["test"]
        test_predicted = apply_tor_threshold(compute_probabilities(model, test_features), arguments.tor_threshold)
        summary_text += "\n" + describe_results("test", test_labels, test_predicted)
        metrics["test_macro_f1"] = macro_f1(test_labels, test_predicted)
        if packet_counts is not None:
            size_table, size_text = describe_by_packet_count("test", test_labels, test_predicted, packet_counts["test"])
            summary_text += "\n" + size_text
            metrics["test_macro_f1_by_packets"] = dict(zip(size_table["packets"], size_table["macro_f1"]))
    return summary_text, metrics


def run_cross_validation(arguments, splits, augmented_mask, packet_counts, channels, model_config, device, run_dir):
    """k-fold CV over train + val, grouped by stream_id so a recording is never in two folds.

    Augmented flows only go into the training side, and only for folds where their original
    stream is also in training, otherwise copies of a val stream would leak into training.
    The out-of-fold predictions are combined to report overall results and to pick the tor threshold.
    (Small caveat: the scaler in the npz was fit on the original train split.)
    """
    train_features, train_labels, train_groups = splits["train"]
    val_features, val_labels, val_groups = splits["val"]
    real_rows = ~augmented_mask

    pool_features = np.concatenate([train_features[real_rows], val_features])
    pool_labels = np.concatenate([train_labels[real_rows], val_labels])
    pool_groups = np.concatenate([train_groups[real_rows], val_groups])
    pool_packet_counts = (None if packet_counts is None
                          else np.concatenate([packet_counts["train"][real_rows], packet_counts["val"]]))
    augmented_features = train_features[augmented_mask]
    augmented_labels = train_labels[augmented_mask]
    augmented_groups = train_groups[augmented_mask]
    print(f"cross-validation pool: {len(pool_labels)} real flows, {len(np.unique(pool_groups))} streams, "
          f"{len(augmented_labels)} augmented flows (train side only)")

    out_of_fold_probabilities = np.zeros((len(pool_labels), len(CLASS_NAMES)), dtype=np.float32)
    fold_of_row = np.zeros(len(pool_labels), dtype=int)
    fold_rows = []
    splitter = StratifiedGroupKFold(n_splits=arguments.cv_folds, shuffle=True, random_state=arguments.seed)
    for fold_number, (fold_train_rows, fold_val_rows) in enumerate(
            splitter.split(pool_features, pool_labels, pool_groups), start=1):
        train_streams = set(pool_groups[fold_train_rows])
        augmented_rows = np.array([stream in train_streams for stream in augmented_groups], dtype=bool)
        fold_train_features = np.concatenate([pool_features[fold_train_rows], augmented_features[augmented_rows]])
        fold_train_labels = np.concatenate([pool_labels[fold_train_rows], augmented_labels[augmented_rows]])
        fold_val_features, fold_val_labels = pool_features[fold_val_rows], pool_labels[fold_val_rows]

        val_counts = dict(zip(CLASS_NAMES, np.bincount(fold_val_labels, minlength=len(CLASS_NAMES)).tolist()))
        print(f"\nfold {fold_number}: train {len(fold_train_labels)} (incl. {augmented_rows.sum()} augmented) "
              f"| val {len(fold_val_labels)} {val_counts}")

        model, history, best_epoch = train_model(
            model_config, channels, (fold_train_features, fold_train_labels), (fold_val_features, fold_val_labels),
            arguments, device, run_dir / f"fold_{fold_number}.pt", log_prefix=f"fold {fold_number} | ")
        history.to_csv(run_dir / f"history_fold_{fold_number}.csv", index=False)
        plot_history(history, run_dir / f"training_curves_fold_{fold_number}.png")

        fold_probabilities = compute_probabilities(model, fold_val_features)
        out_of_fold_probabilities[fold_val_rows] = fold_probabilities
        fold_of_row[fold_val_rows] = fold_number
        fold_predicted = fold_probabilities.argmax(axis=1)
        per_class_f1 = f1_score(fold_val_labels, fold_predicted, labels=range(len(CLASS_NAMES)),
                                average=None, zero_division=0)
        fold_row = {"fold": fold_number, "best_epoch": best_epoch,
                    "macro_f1": macro_f1(fold_val_labels, fold_predicted)}
        for name, score in zip(CLASS_NAMES, per_class_f1):
            fold_row[f"f1_{name}"] = score
        fold_rows.append(fold_row)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(run_dir / "cv_folds.csv", index=False)
    best_threshold, tuned_f1 = tune_tor_threshold(pool_labels, out_of_fold_probabilities)
    tuned_predicted = apply_tor_threshold(out_of_fold_probabilities, best_threshold)

    # save every out-of-fold prediction so we can look at them in a notebook later
    predictions_table = pd.DataFrame({
        "stream_id": pool_groups,
        "fold": fold_of_row,
        "true_label": np.array(CLASS_NAMES)[pool_labels],
    })
    if pool_packet_counts is not None:
        predictions_table["packets_in_window"] = pool_packet_counts
    for index, name in enumerate(CLASS_NAMES):
        predictions_table[f"probability_{name}"] = out_of_fold_probabilities[:, index]
    predictions_table.to_csv(run_dir / "out_of_fold_predictions.csv", index=False)

    # tor recall for each tor recording, to see which ones the model misses
    tor_index = CLASS_NAMES.index("tor")
    tor_rows = pool_labels == tor_index
    tor_streams = (pd.DataFrame({"stream_id": pool_groups[tor_rows],
                                 "caught": tuned_predicted[tor_rows] == tor_index})
                   .groupby("stream_id")["caught"].agg(flows="size", recall="mean")
                   .sort_values("flows", ascending=False))

    summary_text = (
        f"{arguments.cv_folds}-fold cross-validation (grouped by stream)\n"
        f"{folds.round(3).to_string(index=False)}\n\n"
        f"macro-F1 per fold: {folds['macro_f1'].mean():.3f} +/- {folds['macro_f1'].std():.3f}\n\n"
        + describe_results("out-of-fold, argmax", pool_labels, out_of_fold_probabilities.argmax(axis=1))
        + f"\nbest tor threshold on out-of-fold predictions: {best_threshold}\n"
        + describe_results(f"out-of-fold, tor threshold {best_threshold}", pool_labels, tuned_predicted)
        + f"\ntor recall per stream (threshold {best_threshold})\n{tor_streams.round(3).to_string()}\n"
    )
    if pool_packet_counts is not None:
        _, argmax_size_text = describe_by_packet_count(
            "out-of-fold, argmax", pool_labels, out_of_fold_probabilities.argmax(axis=1), pool_packet_counts)
        tuned_size_table, tuned_size_text = describe_by_packet_count(
            f"out-of-fold, tor threshold {best_threshold}", pool_labels, tuned_predicted, pool_packet_counts)
        summary_text += "\n" + argmax_size_text + "\n" + tuned_size_text

    metrics = {
        "cv_macro_f1_mean": float(folds["macro_f1"].mean()),
        "cv_macro_f1_std": float(folds["macro_f1"].std()),
        "out_of_fold_macro_f1_argmax": macro_f1(pool_labels, out_of_fold_probabilities.argmax(axis=1)),
        "best_tor_threshold": best_threshold,
        "out_of_fold_macro_f1_tuned": tuned_f1,
    }
    if pool_packet_counts is not None:
        metrics["out_of_fold_macro_f1_tuned_by_packets"] = dict(zip(tuned_size_table["packets"],
                                                                    tuned_size_table["macro_f1"]))
    return summary_text, metrics


def main():
    arguments = parse_arguments()
    set_seed(arguments.seed)
    device = select_device()

    channels = resolve_channels(arguments.channels)
    splits, augmented_mask, packet_counts = load_split_data(arguments.data_path, channels, arguments.scaler_path)
    model_config = build_model_config(arguments, channels)

    mode = f"cv{arguments.cv_folds}" if arguments.cv_folds else "split"
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{'-'.join(arguments.channels)}"
    run_dir = arguments.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device} | channels: {channels}")
    for split, (features, labels, _) in splits.items():
        counts = dict(zip(CLASS_NAMES, np.bincount(labels, minlength=len(CLASS_NAMES)).tolist()))
        print(f"{split}: {features.shape} {counts}")
    print(f"augmented train rows: {augmented_mask.sum()}")
    if packet_counts is None:
        print(f"note: {arguments.scaler_path} not found, skipping the per-flow-size breakdown")
    parameter_count = sum(parameter.numel() for parameter in FlowCNN(**model_config).parameters())
    print(f"parameters: {parameter_count:,}")

    if arguments.cv_folds:
        if arguments.evaluate_test:
            print("note: --evaluate-test is ignored in cross-validation mode")
        summary_text, metrics = run_cross_validation(arguments, splits, augmented_mask, packet_counts, channels,
                                                     model_config, device, run_dir)
    else:
        summary_text, metrics = run_fixed_split(arguments, splits, packet_counts, channels,
                                                model_config, device, run_dir)

    print("\n" + summary_text)
    (run_dir / "results.txt").write_text(summary_text)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    (run_dir / "run_info.json").write_text(json.dumps(
        {"config": config, "resolved_channels": channels, "device": str(device), "metrics": metrics}, indent=2))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
