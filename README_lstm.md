# LSTM

This part of the project trains an LSTM to classify flows as **non_vpn**, **vpn** or **tor** from the bucketed flow features. The code is in `train_lstm.py` (train + shared helpers) and `tune_lstm.py` (hyper-parameter search). The results are read and plotted in the **LSTM** and **Model Comparison with LSTM** sections of `eda_cleaning.ipynb`; the notebook never trains the LSTM itself.

```
data/final_data.npz --> train_lstm.py --> models/lstm.pt, lstm_history.csv, lstm_seed_runs.csv ---+
                    \-> tune_lstm.py --> models/best_config.json, tuning_results.csv,              +--> eda_cleaning.ipynb
                                         lstm_tuned.pt, tuned_test_scores.csv -------------------+
```

## 1. Set up the environment

Run everything from the repository root. `requirements.txt` is pinned for **Python 3.12**.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Check that it works:

```bash
python -c "import torch, sklearn, pandas; print(torch.__version__, torch.cuda.is_available())"
```

The scripts use the GPU automatically if CUDA is available, otherwise the CPU (the model is small, so the CPU is fine).

## 2. Data

The scripts read `data/final_data.npz` (relative to the **current directory**, so run them from the repo root). It is written by `eda_cleaning.ipynb` (the "save final_data.npz" cell), which saves to the working directory, so move or copy the file into `data/` if you regenerate it. It is already in `data/`.

Each flow has 224 features: 8 channels × 28 log-spaced time buckets over the first 5 s, reshaped to `(28 time steps, 8 channels)` for the LSTM. Splits are grouped by `stream_id`, and only the train split has the augmented Tor flows. `N_BUCKETS`, `BLOCKS` and `SEEDS` in `train_lstm.py` are hard-coded and must match the notebook's `NUM_BUCKETS` and feature order.

## 3. Train the LSTM

The model is a 2-layer LSTM (hidden size 32, dropout 0.2) with mean pooling over time and a linear head. It uses class-weighted cross-entropy and Adam (lr 1e-3), and keeps the epoch with the best validation macro-F1 (early stopping, patience 8).

```bash
python train_lstm.py            # seed 4266 -> models/lstm.pt, models/lstm_history.csv
python train_lstm.py --sweep    # also trains seeds 1-4 and scores val/test -> models/lstm_seed_runs.csv
python train_lstm.py --seed 7   # train with a different seed
```

Run `--sweep` at least once, because the notebook's seed table and model comparison read `lstm_seed_runs.csv`. Importing `train_lstm` (as the notebook does) never trains anything; training only happens when the file is run directly.

## 4. Tune the LSTM (optional, gives the "LSTM (tuned)" row)

`tune_lstm.py` runs a random search over hidden size, layers, bidirectional, dropout, pooling, learning rate, weight decay, batch size, class-weight power and whether to use the Tor augmentation. Every config is scored with 5-fold cross-validation grouped by `stream_id` on train + val (the test set is not used). Stage 1 runs every config with one seed; stage 2 re-runs the best few with more seeds. The final pick uses the one-standard-error rule, which chooses the smallest model within one standard error of the best.

```bash
python tune_lstm.py --smoke                  # tiny run, just to check the pipeline works
python tune_lstm.py --trials 20              # search -> models/tuning_results.csv, models/best_config.json
python tune_lstm.py --final                  # train best_config.json on train+val, score test once
```

Useful options: `--workers N` (run trials in parallel processes), `--fix use_aug=true bidirectional=true` (hold settings fixed), `--max-epochs`, `--folds`, `--top-k`, `--stage2-seeds`. Run `python tune_lstm.py --help` for all of them.

`--final` writes `models/tuned_test_scores.csv` (5 seeds, test only) and `models/lstm_tuned.pt` (the first seed). Run it only after the search has produced `best_config.json`. The search can take a while on a CPU.

## 5. Run the model in the notebook

1. Open `eda_cleaning.ipynb` in VS Code (or Jupyter) and select the **`.venv`** interpreter as the kernel (VS Code: *Select Kernel* → *Python Environments* → `.venv`). For classic Jupyter, register it with `python -m ipykernel install --user --name encrypted-traffic --display-name "Python (.venv)"`.
2. Make sure the training outputs above exist in `models/`.
3. **Restart the kernel**, then run only the LSTM cells, in order, starting from the `# LSTM (2 layers, hidden 32, mean-pool over time)` heading:

| Section | What it does | Needs |
|---|---|---|
| LSTM | loads `models/lstm.pt`, plots the training loss / val macro-F1 curves, prints classification reports and confusion matrices for val and test | `train_lstm.py` |
| Testing the LSTM run on different seeds | shows per-seed scores and mean / std | `train_lstm.py --sweep` |
| Model Comparison with LSTM | compares LogReg, LSTM (5 seeds) and LSTM (tuned, 5 seeds). LogReg is the only model fitted in the notebook (deterministic, one run); both LSTM rows are read from CSVs | `--sweep` and `tune_lstm.py --final` |

The LSTM section is self-contained, so you do not need to run the earlier EDA cells first. The opposite is not true: the LSTM cells import `X_train`, `y_train`, `X_test` and `y_test` from `train_lstm`, which overwrites the notebook's 2-D arrays and string labels with 3-D arrays and integer labels. If you want to re-run the earlier Random Forest / LogReg cells afterwards, restart the kernel first.

## Output files (`models/`)

| File | Written by | Contents |
|---|---|---|
| `lstm.pt`, `lstm_history.csv` | `train_lstm.py` | weights of the best epoch, per-epoch train loss and val F1 |
| `lstm_seed_runs.csv` | `train_lstm.py --sweep` | val and test macro-F1 and per-class F1 for each seed |
| `tuning_results.csv` | `tune_lstm.py` | every config tried with its CV score |
| `best_config.json` | `tune_lstm.py` | chosen config and epoch count |
| `lstm_tuned.pt`, `tuned_test_scores.csv` | `tune_lstm.py --final` | tuned weights and 5-seed test scores |

## Results

Macro-F1 over 5 seeds (mean ± std), from the notebook:

| Model | Val | Test |
|---|---|---|
| LogReg (balanced) | 0.585 | 0.569 |
| LSTM | 0.567 ± 0.007 | 0.419 ± 0.028 |
| LSTM (tuned) | not scored (trained on train + val) | 0.726 ± 0.010 |
