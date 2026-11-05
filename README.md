# Encrypted Network Classification
## DSA4266 Sense-making Case Analysis: Science and Technology

Classifies encrypted network flows as **non_vpn**, **vpn** or **tor** using only flow-level features (packet sizes, counts, direction and timing in the first 5 seconds), without looking at payloads.

# Data
Data used was ISCXVPN2016 and ISCXTor2016 from Canadian Institute for Cybersecurity. Non-Tor dataset was not used as it contained information from ISCXVPN2016. The link to the pages are below.

https://www.unb.ca/cic/datasets/vpn.html

https://www.unb.ca/cic/datasets/tor.html

# How the project works

```
raw .pcap files
   |  parsing.ipynb              parse packets into flows, label by file name   -> data/*_files_AL2.pkl
   v
eda_cleaning.ipynb               clean, bucket into 8 x 28 features, split by stream,
   |                             augment Tor (train only), log1p + scale        -> data/final_data.npz, data/scaler.joblib
   +--> baselines (majority class, LogReg, Random Forest) inside the notebook
   +--> train_lstm.py / tune_lstm.py   LSTM training and tuning                 -> models/lstm/*.pt, models/lstm/*.csv, best_config.json
   +--> cnn_1d.py                      1D CNN training and evaluation           -> outputs/cnn_1d/<run>/
   v
eda_cleaning.ipynb (LSTM sections) reads the saved LSTM results, plots them and compares models
```

1. **Parsing** (`parsing.ipynb`): reads the pcap files, groups packets into flows and labels each flow (`network_type`, `category`, `stream_id`) from the capture file name. The parsed batches are saved as pickles, which are already in `data/`, so you can skip this step unless you have the raw pcaps and want to redo it.
2. **Cleaning and features** (`eda_cleaning.ipynb`, first half): loads the pickles from `data/`, does EDA, turns each flow into 224 features (8 channels × 28 log-spaced time buckets over the first 5 s: bytes out/in, packets out/in, mean/std inter-arrival gap, mean/std packet size), caps flows per stream (200), splits train / val / test **grouped by stream** so no recording appears in two splits, augments Tor flows in the training split only, and applies `log1p` + `StandardScaler` fitted on train. It saves `final_data.npz` and `scaler.joblib`.
3. **Baselines** (`eda_cleaning.ipynb`): majority class, logistic regression and Random Forest, plus the augmentation / undersampling experiments.
4. **LSTM** (`train_lstm.py`, `tune_lstm.py`): trained and tuned from the scripts, then evaluated in the notebook. See [README_lstm.md](README_lstm.md).
5. **1D CNN** (`cnn_1d.py`): trained from the script. See [README_cnn.md](README_cnn.md).

Macro-F1 is the main metric, because accuracy is misleading (always guessing non_vpn already gets about 75% on val).

# Files

| File | Purpose |
|---|---|
| `parsing.ipynb` | pcap -> labelled flows (pickles) |
| `eda_cleaning.ipynb` | EDA, feature engineering, splitting, augmentation, scaling, baselines, LSTM results and model comparison |
| `train_lstm.py` | trains the LSTM (`--sweep` for 5 seeds); also holds the model, data loading and scoring helpers |
| `tune_lstm.py` | random search with grouped CV for the LSTM (`--final` scores the chosen config on test) |
| `cnn_1d.py` | trains / cross-validates the 1D CNN |
| `README_lstm.md`, `README_cnn.md` | details for each model |
| `requirements.txt` | pinned dependencies (Python 3.12) |
| `data/` | parsed pickles, `final_data.npz`, `scaler.joblib` |
| `models/` | saved weights and result CSVs, one folder per model (`lstm/`, `cnn_1d/`) |

# Setup

Run everything from the repository root. Python 3.12 is recommended (the versions in `requirements.txt` were tested with it).

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Select the `.venv` interpreter as the kernel when opening the notebooks.

# What to run

The repository should already contain `data/final_data.npz`, so you can go straight to step 2.

| Step | Command / file | Produces |
|---|---|---|
| 1. Rebuild the dataset (optional) | run `parsing.ipynb` (set your pcap paths), then the cleaning, split, augmentation, scaling and save cells of `eda_cleaning.ipynb`. The notebook saves `final_data.npz` and `scaler.joblib` straight into `data/` | `data/final_data.npz`, `data/scaler.joblib` |
| 2. Train the LSTM | `python train_lstm.py --sweep` | `models/lstm/lstm.pt`, `lstm_history.csv`, `lstm_seed_runs.csv` |
| 3. Tune the LSTM (optional) | `python tune_lstm.py --trials 20`, then `python tune_lstm.py --final` | `models/lstm/tuning_results.csv`, `best_config.json`, `lstm_tuned.pt`, `tuned_test_scores.csv` |
| 4. Train the 1D CNN | `python cnn_1d.py --no-class-weights --train-on-train-val --epochs 36 --seed 4266` (more options in `README_cnn.md`) | `outputs/cnn_1d/<run>/` |
| 5. Generate the results | open `eda_cleaning.ipynb`, run the baseline, CNN and LSTM cells.

Notes:

- `train_lstm.py` and `tune_lstm.py` load `data/final_data.npz` relative to the current directory, so run them from the repo root.
- The LSTM cells in the notebook import arrays from `train_lstm`, which overwrites the notebook's own `X_train` / `y_train` etc. Restart the kernel before re-running earlier baseline cells.
- Model comparison in the notebook shows LogReg, the LSTM (5 seeds) and the tuned LSTM (5 seeds). The LSTM rows are read from the CSVs written in steps 2 and 3, so run those first.
