import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np  # keep before torch (OMP libiomp5md clash on Windows/anaconda)
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

import train_lstm as T
from train_lstm import BLOCKS, CLASSES, MODEL_DIR, RNNClassifier, SEEDS, predict, score_row, to_seq, train_torch

DATA_PATH = T.ROOT / "data" / "final_data.npz"
RESULTS_PATH = MODEL_DIR / "tuning_results.csv"
BEST_PATH = MODEL_DIR / "best_config.json"
FINAL_PATH = MODEL_DIR / "tuned_test_scores.csv"
SMOOTH = 3   # epochs averaged when picking the epoch count

# the settings used so far, always evaluated as the reference
BASELINE = dict(hidden=32, layers=2, bidirectional=False, dropout=0.2, pooling="mean", zero_empty=False,
                add_mask=False, lr=1e-3, weight_decay=0.0, batch_size=256, class_weight_power=1.0, use_aug=True)


def parse_fixes(items):
    """['use_aug=true', 'pooling=max'] -> {'use_aug': True, 'pooling': 'max'}, checked against BASELINE's types."""
    fixes = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in BASELINE:
            raise SystemExit(f"--fix expects name=value with name in {sorted(BASELINE)}; got '{item}'")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        kind = type(BASELINE[key])
        if kind is bool and not isinstance(value, bool) or kind is not bool and isinstance(value, bool):
            raise SystemExit(f"--fix {key} needs a {kind.__name__}, got '{raw}'")
        fixes[key] = kind(value)
    if fixes.get("pooling") == "masked_mean" and fixes.get("add_mask") is False:
        raise SystemExit("pooling=masked_mean needs add_mask=true")
    return fixes


def sample_config(rng, fixes=None):
    cfg = dict(
        hidden=int(rng.choice([16, 32, 64])),
        layers=int(rng.choice([1, 2])),
        bidirectional=bool(rng.choice([False, True])),
        dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5])),
        pooling=str(rng.choice(["mean", "max", "masked_mean"])),
        zero_empty=bool(rng.choice([False, True])),   # set empty slices to 0 instead of the scaled constant
        add_mask=bool(rng.choice([False, True])),     # extra "slice is non-empty" input channel
        lr=float(10 ** rng.uniform(-3.5, -2.3)),
        weight_decay=float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        batch_size=int(rng.choice([128, 256])),
        class_weight_power=float(rng.choice([0.5, 0.75, 1.0])),   # 1.0 = fully balanced class weights
        use_aug=bool(rng.choice([False, True])),
    )
    fixes = fixes or {}
    cfg.update(fixes)
    if cfg["pooling"] == "masked_mean":   # it needs the non-empty channel
        if fixes.get("add_mask") is False:
            cfg["pooling"] = "mean"
        else:
            cfg["add_mask"] = True
    return cfg


class Pool:
    """train + val flows (train includes the appended augmented rows) as sequences."""
    def __init__(self):
        d = np.load(DATA_PATH)
        label_id = {c: i for i, c in enumerate(CLASSES)}
        y_train = d["y_train"]
        if "n_real_train" in d.files:
            n_real = int(d["n_real_train"])
        else:   # augmented rows are appended at the end and are all Tor
            n_real = len(y_train)
            while n_real > 0 and y_train[n_real - 1] == "tor":
                n_real -= 1
            print(f"n_real_train not in {DATA_PATH.name}; assuming the last {len(y_train) - n_real} "
                  f"train rows are augmented")
        self.X3 = to_seq(np.concatenate([d["X_train"], d["X_val"]]))
        self.y = np.array([label_id[v] for v in np.concatenate([y_train, d["y_val"]])])
        self.g = np.concatenate([d["groups_train"], d["groups_val"]]).astype(str)
        self.is_aug = np.zeros(len(self.y), dtype=bool)
        self.is_aug[n_real:len(y_train)] = True
        # scaled value of an empty slice: packets_out and packets_in at their minimum
        self.pkt = [BLOCKS.index("packets_out"), BLOCKS.index("packets_in")]
        self.empty_value = self.X3[:, :, self.pkt].min(axis=(0, 1))

    def inputs(self, X3, cfg):
        """Apply the per-config input options (zero empty slices, add a non-empty channel)."""
        empty = (X3[:, :, self.pkt] <= self.empty_value + 1e-5).all(axis=-1)
        out = X3.copy()
        if cfg["zero_empty"]:
            out[empty] = 0.0
        if cfg["add_mask"]:
            out = np.concatenate([out, (~empty)[..., None].astype(np.float32)], axis=2)
        return np.ascontiguousarray(out, dtype=np.float32)


def make_folds(pool, n_folds, seed):
    real = np.flatnonzero(~pool.is_aug)
    aug = np.flatnonzero(pool.is_aug)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, te in sgkf.split(real, pool.y[real], pool.g[real]):
        held, train_real = real[te], real[tr]
        assert not set(pool.g[held]) & set(pool.g[train_real])
        train_aug = aug[~np.isin(pool.g[aug], np.unique(pool.g[held]))]
        folds.append(dict(train_real=train_real, train_aug=train_aug, held=held))
    return folds


def make_model(cfg):
    n_in = len(BLOCKS) + int(cfg["add_mask"])
    dropout = cfg["dropout"] if cfg["layers"] > 1 else 0.0   # torch ignores/warns on dropout with 1 layer
    return lambda: RNNClassifier("lstm", n_in=n_in, hidden=cfg["hidden"], layers=cfg["layers"], dropout=dropout,
                                 bidirectional=cfg["bidirectional"], pooling=cfg["pooling"])


def n_params(cfg):
    return sum(p.numel() for p in make_model(cfg)().parameters())


def train_kwargs(cfg):
    return dict(lr=cfg["lr"], batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
                class_weight_power=cfg["class_weight_power"])


def evaluate(cfg, pool, folds, seeds, max_epochs):
    """Held-out curves, shape (folds*seeds, epochs, 1 + len(CLASSES)): macro-F1 then per-class F1."""
    X = pool.inputs(pool.X3, cfg)
    cols = ["val_macro_f1", *[f"f1_{c}" for c in CLASSES]]
    curves = []
    for seed in seeds:
        for f in folds:
            tr = np.concatenate([f["train_real"], f["train_aug"]]) if cfg["use_aug"] else f["train_real"]
            _, hist = train_torch(make_model(cfg), X[tr], pool.y[tr], X[f["held"]], pool.y[f["held"]], seed=seed,
                                  epochs=max_epochs, patience=max_epochs + 1, restore_best=False,
                                  **train_kwargs(cfg))
            curves.append(hist[cols].to_numpy())
    return np.stack(curves)


def summarise(curves):
    """Pick the epoch count from the fold-averaged, smoothed macro-F1 curve and report it with its spread."""
    smooth = min(SMOOTH, curves.shape[1])
    mean_curve = curves.mean(axis=0)[:, 0]
    smoothed = np.convolve(mean_curve, np.ones(smooth) / smooth, mode="valid")
    k = int(smoothed.argmax())
    window = curves[:, k:k + smooth, :].mean(axis=1)   # (runs, 1 + classes) around the chosen epoch
    runs = len(window)
    return {"epochs": k + smooth // 2 + 1, "score": window[:, 0].mean(), "std": window[:, 0].std(ddof=1) if runs > 1 else np.nan,
            "se": window[:, 0].std(ddof=1) / np.sqrt(runs) if runs > 1 else np.nan, "runs": runs,
            **{f"f1_{c}": window[:, 1 + i].mean() for i, c in enumerate(CLASSES)}}


_STATE = {}   # per-process data, set by _init (in the main process and in every worker)


def _init(n_folds, fold_seed, max_epochs, threads):
    if threads:
        torch.set_num_threads(threads)
    pool = Pool()
    _STATE.update(pool=pool, folds=make_folds(pool, n_folds, fold_seed), max_epochs=max_epochs)


def _run(cfg, seeds):
    started = time.time()
    curves = evaluate(cfg, _STATE["pool"], _STATE["folds"], seeds, _STATE["max_epochs"])
    return curves, (time.time() - started) / 60


def run_jobs(jobs, workers, init_args):
    """jobs: {key: (cfg, seeds)}. Yields (key, curves, minutes) as each finishes (in parallel if workers > 1)."""
    if workers <= 1:
        for key, (cfg, seeds) in jobs.items():
            yield (key, *_run(cfg, seeds))
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=init_args) as executor:
        futures = {executor.submit(_run, cfg, seeds): key for key, (cfg, seeds) in jobs.items()}
        for future in as_completed(futures):
            yield (futures[future], *future.result())


def search(args):
    fixes = parse_fixes(args.fix)
    # all cores split between workers; a single worker keeps torch's default
    threads = args.threads or (max(1, (os.cpu_count() or 8) // 2 // args.workers) if args.workers > 1 else 0)
    init_args = (args.folds, args.seed, args.max_epochs, threads)
    _init(*init_args)
    pool, folds = _STATE["pool"], _STATE["folds"]
    print(f"{len(pool.y)} flows ({int((~pool.is_aug).sum())} real + {int(pool.is_aug.sum())} augmented), "
          f"{len(set(pool.g[~pool.is_aug]))} streams, {args.folds} folds, device {T.DEVICE}, "
          f"{args.workers} worker(s), fixed: {fixes or 'nothing'}")
    for i, f in enumerate(folds):
        print(f"  fold {i}: held-out {np.bincount(pool.y[f['held']], minlength=len(CLASSES)).tolist()} "
              f"{CLASSES}, {len(np.unique(pool.g[f['held']]))} streams")

    rng = np.random.default_rng(args.seed)
    cfgs = [BASELINE] + [sample_config(rng, fixes) for _ in range(args.trials)]   # trial 0 is the reference
    curves, rows = {}, []

    def record(i, stage, runs_curves, minutes):
        row = {"trial": i, "stage": stage, **cfgs[i], **summarise(runs_curves), "params": n_params(cfgs[i]),
               "minutes": minutes}
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
        print(f"[stage {stage}] trial {i:>3} score {row['score']:.3f} (se {row['se']:.3f}) epochs {row['epochs']:>3} "
              f"tor F1 {row['f1_tor']:.3f} params {row['params']:>6} {row['minutes']:.1f} min | {cfgs[i]}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    jobs = {i: (cfg, SEEDS[:1]) for i, cfg in enumerate(cfgs)}
    for i, c, minutes in run_jobs(jobs, args.workers, init_args):
        curves[i] = c
        record(i, 1, c, minutes)

    stage1 = pd.DataFrame(rows).set_index("trial")
    top = stage1.drop(index=0).sort_values("score", ascending=False).head(args.top_k).index.tolist()
    extra_seeds = SEEDS[1:args.stage2_seeds]
    if extra_seeds:
        jobs = {i: (cfgs[i], extra_seeds) for i in [0, *top]}
        for i, c, minutes in run_jobs(jobs, args.workers, init_args):
            curves[i] = np.concatenate([curves[i], c])
            record(i, 2, curves[i], minutes)
    else:
        for i in [0, *top]:
            record(i, 2, curves[i], 0.0)

    final = pd.DataFrame(rows).query("stage == 2")
    best = final.loc[final["score"].idxmax()]
    close = final[final["score"] >= best["score"] - best["se"]]       # one-standard-error rule
    chosen = close.loc[close["params"].idxmin()]
    out = {k: (chosen[k].item() if hasattr(chosen[k], "item") else chosen[k]) for k in [*BASELINE, "epochs"]}
    BEST_PATH.write_text(json.dumps(out, indent=2))
    base = final[final["trial"] == 0].iloc[0]
    print(f"\nbaseline : {base['score']:.3f} +/- {base['se']:.3f} (tor F1 {base['f1_tor']:.3f})")
    print(f"best     : trial {int(best['trial'])} {best['score']:.3f} +/- {best['se']:.3f}")
    print(f"chosen   : trial {int(chosen['trial'])} {chosen['score']:.3f} +/- {chosen['se']:.3f} -> {BEST_PATH}")
    print(json.dumps(out))


def final(args):
    """Train the chosen config on the whole pool for its fixed epoch count; test is scored once."""
    cfg = json.loads(BEST_PATH.read_text())
    epochs = cfg.pop("epochs")
    pool = Pool()
    train = np.flatnonzero(pool.is_aug & bool(cfg["use_aug"]) | ~pool.is_aug)
    X, X_test = pool.inputs(pool.X3, cfg), pool.inputs(T.X_test, cfg)
    rows = []
    for seed in SEEDS:
        model, _ = train_torch(make_model(cfg), X[train], pool.y[train], seed=seed, epochs=epochs, **train_kwargs(cfg))
        rows.append(score_row(T.y_test, predict(model, X_test), seed=seed, split="test"))
        print(f"seed {seed} done", flush=True)
        if seed == SEEDS[0]:
            torch.save(model.state_dict(), MODEL_DIR / "lstm_tuned.pt")
    scores = pd.DataFrame(rows)
    scores.to_csv(FINAL_PATH, index=False)
    print(scores.round(3).to_string(index=False))
    print("tuned test macro-F1: mean {:.3f} std {:.3f}".format(scores["macro_f1"].mean(), scores["macro_f1"].std()))
    if T.SEED_RUNS_PATH.exists():
        old = pd.read_csv(T.SEED_RUNS_PATH).query("split == 'test'")["macro_f1"]
        print("original LSTM test macro-F1: mean {:.3f} std {:.3f}".format(old.mean(), old.std()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="random configs besides the baseline")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=4, help="configs re-run with more seeds in stage 2")
    parser.add_argument("--stage2-seeds", type=int, default=3, help="total seeds per config after stage 2")
    parser.add_argument("--seed", type=int, default=0, help="seed for fold assignment and config sampling")
    parser.add_argument("--fix", nargs="+", metavar="NAME=VALUE",
                        help="keep these settings fixed in every trial, e.g. use_aug=true bidirectional=true")
    parser.add_argument("--workers", type=int, default=1, help="trials to run in parallel processes")
    parser.add_argument("--threads", type=int, default=0, help="torch threads per worker (default: split the cores)")
    parser.add_argument("--final", action="store_true", help="train best_config.json on train+val and score test")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = parser.parse_args()
    if args.smoke:
        args.trials, args.folds, args.max_epochs, args.top_k, args.stage2_seeds = 2, 3, 3, 1, 2
    final(args) if args.final else search(args)
