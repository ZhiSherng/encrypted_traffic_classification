# 1D CNN

This part of the project trains a 1D CNN to classify flows as **non_vpn**, **vpn** or **tor** using only the bucketed flow features (packet sizes, counts, direction and timing). Everything is in `cnn_1d.py`.

Final result: **test macro-F1 of 0.745**, and **0.731 ± 0.018** in cross-validation (5-fold, repeated with 5 seeds). Most of the remaining errors come from flows that only have 1–2 packets, which is explained further down.

## Data

The model reads `data/final_data.npz` from `eda_cleaning.ipynb`, so the cleaning, bucketing, splitting, augmentation and scaling are the same as the rest of the group's pipeline.

- Each flow has 224 features: 8 channels × 28 log-spaced time buckets covering the first 5 seconds of the flow. The channels are bytes out/in, packets out/in, mean/std of inter-arrival gap, and mean/std of packet size.
- For the CNN we reshape each flow into an (8, 28) array, so the convolution moves along time.
- The npz is already log1p-transformed and scaled, so the script doesn't scale anything again.

| Split | Flows | non_vpn | vpn | tor |
|---|---|---|---|---|
| train | 10,829 | 6,939 | 2,235 | 1,655 |
| val | 1,599 | 1,196 | 342 | 61 |
| test | 2,172 | 1,552 | 541 | 79 |

The training Tor flows include the augmented flows from `eda_cleaning.ipynb`. The augmentation grows Tor to at most 5× its original size, so the 331 real training Tor flows became 1,655, meaning **1,324 of the training Tor flows are synthetic**. They are appended at the end of train, but the npz doesn't mark them, so `cnn_1d.py` identifies them as the last 1,324 rows. This matters for cross-validation (see below).

## Model

```
input (8 channels x 28 buckets)
-> Conv1d(32, k=3) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)    # 28 -> 14 steps
-> Conv1d(64, k=3) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)    # 14 -> 7 steps
-> Flatten -> Linear(64) -> ReLU -> Dropout(0.3) -> Linear(3)
```

- About 36k parameters, which is quite small, but the dataset is also small.
- We flatten at the end instead of using global pooling. Because the buckets are log-spaced, *when* something happens in the flow matters (e.g. the first few milliseconds vs later on), and global pooling would lose that.
- Training setup: AdamW (lr 1e-3, weight decay 1e-4), batch size 128, cross-entropy loss, up to 100 epochs with early stopping (patience 15) on validation macro-F1. The weights from the best epoch are kept.
- Seed is 4266 like the rest of the project.

## How we evaluated

**Metric.** Macro-F1 is the main metric, since accuracy is misleading here (always guessing non_vpn already gets ~75% on val).

**Why we used cross-validation.** The fixed validation set only has 61 Tor flows, and 41 of them come from one recording (`torYoutube1`). So the val Tor score mostly measures how well the model does on that single recording, and it changes a lot between runs. To get a more reliable number, the script can run k-fold cross-validation (`--cv-folds 5`):

- train + val (only the real flows) are combined and split into folds with `StratifiedGroupKFold`, grouped by `stream_id`, so a recording never appears in both training and validation
- the augmented Tor flows are only added to the training side, and only for folds where their original recording is also in training. Otherwise copies of a validation recording would leak into training
- this gives 11,104 real flows from 134 recordings, and every Tor recording gets evaluated once

The test set was not used at all until the very last run.

**Tor threshold.** The script can also tune a threshold for Tor (`--tor-threshold`): a flow is only predicted as Tor if P(tor) is above the threshold, otherwise it goes to non_vpn or vpn. The threshold is picked on the out-of-fold predictions from CV.

**Breakdown by flow size.** Every run also reports results split by how many packets the flow has inside the 5 s window (1–2, 3–10, 11+). We added this after finding that small flows are the main problem.

## Experiments

### Stage 1: experiments 

5-fold CV, default settings (these runs still used class weights except where noted).

The channel experiments check which kind of information the model actually relies on. Each run only gives the CNN some of the 8 channels (`--channels`):

- **all channels**: all 8 channels, which is the normal setup
- **volume**: how much traffic there is in each bucket, i.e. bytes out, bytes in, packets out and packets in (4 channels)
- **size**: the mean and standard deviation of packet size in each bucket (2 channels)
- **timing**: the mean and standard deviation of the gap between packets in each bucket (2 channels)
- **size + timing**: the size and timing channels together (4 channels), so everything except volume

The first row (all channels, with class weights and default dropout / weight decay) is the reference run. Every other row changes one thing from it, and all runs use the same seed, so they are evaluated on the same folds.

| Experiment | CV macro-F1 |
|---|---|
| all channels | 0.660 ± 0.060 |
| volume only (bytes + packets) | 0.648 ± 0.053 |
| size + timing | 0.632 ± 0.039 |
| size only | 0.601 ± 0.045 |
| timing only | 0.592 ± 0.032 |
| all channels, **no class weights** | **0.732 ± 0.112** |
| all channels, no dropout / weight decay | 0.676 ± 0.062 |

- **Volume carries most of the information.** It's almost as good as all 8 channels together. Size and timing on their own are weaker, but combining them helps, and using all channels is still the best (better than volume on 4 of 5 folds).
- **Class weights made things worse.** Removing them improved every single fold. Our explanation is that the training set is already rebalanced by the Tor augmentation (Tor is ~15% of train but ~4% of the real data), so adding class weights on top made the model predict Tor far too often.
- **Regularisation didn't really matter.** Turning it off was slightly better on all folds, but only by 0.005–0.025.

### Stage 2: hyperparameter tuning

Since Stage 1 showed that removing class weights helps, the control for this stage is the "no class weights" run from Stage 1 (CV macro-F1 0.732 ± 0.112): all 8 channels, no class weights, learning rate 1e-3, conv channels 32-64, kernel size 3, dropout 0.3, weight decay 1e-4, batch size 128, seed 4266. Each run below changes only one of these settings and is compared against 0.732:

| Setting | Values tried | CV macro-F1 range |
|---|---|---|
| learning rate | 3e-4, 3e-3 | 0.728 – 0.747 |
| conv channels | 16-32, 64-128, 32-64-128 | 0.731 – 0.737 |
| kernel size | 5, 7 | 0.740 – 0.743 |
| dropout | 0.1, 0.5 | 0.735 – 0.738 |
| weight decay | 0, 1e-3 | 0.739 – 0.744 |
| batch size | 64, 256 | 0.734 – 0.738 |

Everything landed between 0.728 and 0.747. The differences (at most 0.015) are much smaller than the spread between folds (around ±0.11), and no setting was better on more than 4 of 5 folds. So we kept the default settings. A smaller model (16-32) did just as well as the default, and bigger ones didn't help, which suggests the model size isn't what's holding it back.

### Stage 3: seed check

To make sure the result wasn't down to one lucky split, we reran the chosen setup and the most consistent Stage 2 alternative (weight decay 1e-3) with 5 different seeds. Changing the seed also reshuffles which recordings go into which fold.

| | default (chosen) | weight decay 1e-3 |
|---|---|---|
| CV macro-F1 | **0.731 ± 0.018** | 0.728 ± 0.018 |
| F1 non_vpn / vpn / tor | 0.933 / 0.806 / 0.454 | 0.933 / 0.808 / 0.444 |
| macro-F1 on 1–2 / 3–10 / 11+ packet flows | 0.61 / 0.76 / 0.84 | 0.60 / 0.76 / 0.85 |

The result is stable across seeds, and weight decay 1e-3 is not actually better (it was slightly worse on 4 of 5 seeds).

**About the Tor threshold.** With class weights, tuning the threshold helped a lot (best threshold 0.88, out-of-fold macro-F1 went from 0.656 to 0.697, and false Tor predictions on non_vpn dropped from 530 to 57). Without class weights it stopped helping: the tuned and plain argmax results were the same (0.709 vs 0.709), and the "best" threshold jumped between 0.50 and 0.83 depending on the seed. So the threshold was really just fixing the problem caused by class weights, and the final model uses plain argmax.

## Final model

Settings: all channels, no class weights, everything else default, seed 4266.

```bash
python cnn_1d.py --no-class-weights --evaluate-test
```

**Test set:**

| | precision | recall | F1 |
|---|---|---|---|
| non_vpn | 0.914 | 0.966 | 0.939 |
| vpn | 0.919 | 0.835 | 0.875 |
| tor | 0.625 | 0.316 | 0.420 |
| **macro-F1** | | | **0.745** |

| Packets in window | Share of test flows | Macro-F1 | Tor F1 |
|---|---|---|---|
| 1–2 | 63% | 0.615 | 0.000 |
| 3–10 | 18% | 0.725 | 0.593 |
| 11+ | 19% | 0.831 | 0.791 |

The test score (0.745) is in line with the CV estimate (0.731 ± 0.018), so the tuning doesn't seem to have overfitted. Validation was a bit higher (0.760), which is expected since validation was used to choose the best epoch.

## What limits the model

**Most errors are on tiny flows.** Most flows in this dataset only have 1–2 packets in the first 5 seconds (the median is 2 packets for every class). A 2-packet Tor flow and a 2-packet non_vpn flow look almost the same, so there isn't much any model can do with them. In the CV results, 63% of Tor flows had 1–2 packets and the model only caught 3% of those, while on Tor flows with more than 10 packets recall was 60–75%. This also explains why some big Tor recordings look like complete failures: 75% of `Torrent01` flows and 92% of the Tor YouTube/Vimeo flows have only 1–2 packets.

**Background traffic is still in the data.** A large part of the raw flows are LLMNR multicast (Windows name lookups) and DNS, which are 1–2 packet flows that show up under all three labels. We discussed removing them as a group but decided to keep the data as it is, so they are part of the reason the small-flow results are low.

**Few Tor recordings.** There are only 331 real Tor flows in train, and val/test Tor are each dominated by one YouTube recording (41 of 61 and 56 of 79 flows). The CV results are more reliable than the single val/test numbers for Tor because of this.

**Only the first 5 seconds.** The buckets only cover the first 5 s of each flow, so long flows are judged on their opening only.

**One fold is always hard.** In every CV run, the fold containing `Torrent01` scored around 0.55–0.59, no matter which settings we used.

## Running the code

```bash
pip install -r requirements.txt

python cnn_1d.py                                  # normal run (train/val)
python cnn_1d.py --cv-folds 5 --no-class-weights  # cross-validation
python cnn_1d.py --channels size timing           # use only some channels
python cnn_1d.py --no-class-weights --evaluate-test   # final run, only do this once
```

Run `python cnn_1d.py --help` to see all the options. Each run saves to its own folder in `outputs/cnn_1d/` (gitignored):

- `results.txt`: classification report, confusion matrix and the flow-size breakdown
- `run_info.json`: the settings used and the main metrics
- `best_model.pt`: the model (in CV mode, one `fold_k.pt` per fold)
- `history.csv` and `training_curves.png`: loss and F1 per epoch
- CV runs also save `cv_folds.csv` and `out_of_fold_predictions.csv`

A CV run takes about a minute on a MacBook (it uses the Mac GPU through MPS automatically).

**Using the trained model in a notebook:**

```python
import numpy as np
from cnn_1d import load_model, predict

model, checkpoint = load_model("outputs/cnn_1d/20261028_195759_split_all/best_model.pt")
data = np.load("data/final_data.npz")
predicted_labels, probabilities = predict(model, checkpoint, data["X_test"])
```

`predict` expects the already-scaled features from the npz. For new raw flows, apply `np.log1p` and `data/scaler.joblib` first.

## Possible next steps

- LSTM / GRU and CNN + LSTM models, to compare with the CNN as planned
- using the first N raw packets as input instead of the buckets, for the "how early can we classify" (5/10/20/50 packets) experiment
- comparing against the group's statistical baseline in one table
