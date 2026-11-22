# Random forest: detecting label switches from headset motion

## Summary

- The goal is to flag frames within 30 frames (1 s) of a change in the `label` code in `../Data/HeadsetFinal`. Any change of code counts as a switch.
- Range-of-motion clusters give a readable mapping from motion to switch likelihood:
  - Still chunks contain switches 0.3–0.4× as often as average.
  - Chunks with large head turns contain them up to 6× as often.
  - Some pairs of consecutive clusters contain them up to 11× as often.
- No cluster is mostly near-switch. Even the most switch-prone cluster (10 s chunks with about a 70° turn) contains a switch only 23% of the time.
- As a detector, the forest is only somewhat better than chance:
  - The best configuration (16 clusters plus hand-built window features) has a cross-validation ROC-AUC of 0.775.
  - On the two held-out students it catches 5 of 47 switches within ±1 s, with about 42 false alarms per hour.
- More clusters (32 instead of 16) did not help.

## Data and labels

- **Data:** 40 students, resampled to 30 Hz with `resample_session_to_30hz` from `preprocessing/dataLoaders.py`. After resampling there are 4.87M frames, and 30 frames is exactly 1 s in every file.
- **Recording gaps:** gaps longer than 1 s split a session into separate segments. Windows never cross a gap.
- **Switches:** 683 in total, 636 in the training students and 47 in the test students.
- **Positive label:** a frame is positive when a switch is within 30 frames on either side (61 frames per switch). That is 0.81% of training frames and 1.35% of test frames.
- **Train/test split:** students 31 and 41 are held out for testing, the same as `HEADSET_FINAL_TEST_SUBJECTS` in `preprocessing/config.py`. Every other student is used for training. Training frames are subsampled to every 3rd frame; test frames are all kept.

## Normalization

Pose is normalized as in `ConvTran/dataset.py`: every window is expressed relative to its first frame. Position becomes a translation from [0, 0, 0], and rotation becomes `inverse(q0) * q_t`. No feature depends on where a student stood or which way they faced.

This was verified by moving a whole session to a new position and starting orientation. No feature changed by more than about 4e-9.

## Motion clusters

1. Each segment is cut into 1 s, 3 s and 10 s chunks, each overlapping the previous one by half.
2. Each chunk is centered on its first pose and described by eight range-of-motion values:
   - horizontal reach
   - vertical range
   - pitch range
   - yaw range
   - roll range
   - peak turn away from the starting orientation
   - distance traveled
   - total rotation
3. Gaze is left out of the default (headset) clusters, because students 7, 8, 10 and 17 have no gaze recorded. The `--gaze` runs drop those students and add three gaze descriptors per chunk: path length, peak step, and valid-share.
4. The values are log-scaled and standardized, then clustered with k-means (16 clusters per chunk length). K-means is fitted on training students only.
5. Clusters are numbered by activity, from 0 (stillest) to 15 (most active).

### Cluster profiles, 10 s chunks (training students)

"Contains a switch" is the fraction of chunks in the cluster that contain at least one switch. "Lift" is that rate divided by the rate for all 10 s chunks (3.8%).

| Cluster | Share | Reach (m) | Pitch range | Yaw range | Peak turn | Distance traveled (m) | Contains a switch | Lift | Most common code |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 14.1% | 0.003 | 1° | 1° | 1° | 0.06 | 1.5% | 0.39 | 3 (73%) |
| 1 | 18.0% | 0.010 | 3° | 3° | 4° | 0.11 | 1.1% | 0.29 | 3 (88%) |
| 2 | 15.8% | 0.020 | 6° | 6° | 7° | 0.16 | 1.4% | 0.37 | 3 (90%) |
| 3 | 7.9% | 0.027 | 6° | 14° | 13° | 0.17 | 2.2% | 0.60 | 3 (93%) |
| 4 | 7.3% | 0.061 | 9° | 8° | 10° | 0.27 | 2.8% | 0.76 | 3 (89%) |
| 5 | 6.1% | 0.038 | 8° | 11° | 15° | 0.25 | 2.2% | 0.58 | 3 (90%) |
| 6 | 4.1% | 0.038 | 21° | 11° | 20° | 0.27 | 3.6% | 0.95 | 3 (75%) |
| 7 | 4.8% | 0.061 | 13° | 28° | 27° | 0.31 | 5.8% | 1.53 | 3 (84%) |
| 8 | 4.5% | 0.137 | 15° | 15° | 17° | 0.50 | 6.7% | 1.79 | 3 (85%) |
| 9 | 2.5% | 0.075 | 15° | 17° | 34° | 0.35 | 2.6% | 0.69 | 3 (81%) |
| 10 | 4.3% | 0.138 | 15° | 48° | 47° | 0.48 | 8.6% | 2.29 | 3 (82%) |
| 11 | 4.2% | 0.123 | 33° | 26° | 37° | 0.53 | 12.1% | 3.22 | 3 (69%) |
| 12 | 0.5% | 0.753 | 29° | 25° | 34° | 2.06 | 14.2% | 3.78 | 6 (32%) |
| 13 | 2.7% | 0.222 | 37° | 70° | 73° | 0.84 | 23.5% | 6.25 | 3 (46%) |
| 14 | 1.3% | 0.761 | 70° | 96° | 107° | 2.26 | 14.8% | 3.95 | 6 (51%) |
| 15 | 1.8% | 1.167 | 13° | 94° | 71° | 122.4 | 2.7% | 0.72 | 3 (98%) |

What the table shows:

- **Still clusters (0–2)** cover 48% of the time and rarely contain a switch.
- **Turning clusters (7, 10, 11, 13)** and **walking-while-turning clusters (12, 14)** are switch-prone, with a lift of 1.5 or more. Cluster 13, a large turn with a short step, is the strongest at 6.25×.
- **Cluster 15 is tracking glitches**, not real motion: 122 m traveled in 10 s is physically impossible. It should be filtered out before clustering in future runs.
- **Label code 3 dominates almost every cluster.** Mapping clusters to label codes directly would assign code 3 nearly everywhere. Only the walking clusters (12, 14) lean towards code 6.

The 1 s and 3 s profiles show the same ordering. They are in `output_clusters16/cluster_profiles.csv`.

### Cluster transitions

Two back-to-back chunks that don't overlap form a pair. A pair counts as near a switch when a switch falls within ±1 s of the boundary between the chunks. The table lists the pairs with the highest lift that occur at least 50 times. Everything is in `output_clusters16/cluster_transitions.csv`.

| Chunk length | From cluster | To cluster | Times seen | Near a switch | Lift |
|---|---|---|---|---|---|
| 1 s | 10 | 13 | 151 | 6.6% | 8.3 |
| 1 s | 6 | 13 | 100 | 5.0% | 6.2 |
| 1 s | 12 | 13 | 630 | 4.6% | 5.8 |
| 3 s | 13 | 5 | 117 | 9.4% | 11.6 |
| 3 s | 12 | 6 | 158 | 8.2% | 10.2 |
| 3 s | 13 | 3 | 110 | 7.3% | 9.0 |
| 10 s | 13 | 14 | 71 | 8.5% | 10.6 |
| 10 s | 13 | 11 | 109 | 7.3% | 9.2 |
| 10 s | 10 | 3 | 58 | 6.9% | 8.7 |

The strongest pattern is a large turn followed by a calmer chunk. Even so, fewer than 1 in 10 of these boundaries is near a switch.

## Forest results

### Setup

- **Forest:** 300 trees, minimum 5 samples per leaf, `class_weight='balanced_subsample'`. Up to 20 negatives are kept per positive when fitting.
- **Turning probabilities into detections:** a detected switch is a probability peak above the threshold. Peaks within 30 frames of each other are merged into one.
- **Scoring:** a detection is correct when it is within ±30 frames of a real switch. Each switch can be matched only once.
- **Threshold:** chosen to maximize this switch-level F1 in 5-fold leave-students-out cross-validation on the training students. The test students are never used to choose it.

### Comparison

| Inputs | Cross-validation ROC-AUC | Cross-validation average precision | Cross-validation switches caught | Test ROC-AUC | Test switches caught | Test detections | Test precision | Test F1 | Test false alarms per hour |
|---|---|---|---|---|---|---|---|---|---|
| Hand-built window features (83) | 0.751 | 0.029 | 117 / 636 | 0.643 | 4 / 47 | 98 | 4.1% | 0.055 | 48 |
| 16 clusters (147 features) | 0.755 | 0.026 | 108 / 636 | 0.567 | 7 / 47 | 150 | 4.7% | 0.071 | 74 |
| 32 clusters (291 features) | 0.761 | 0.026 | 90 / 636 | 0.592 | 5 / 47 | 176 | 2.8% | 0.045 | 88 |
| 16 clusters + window features | **0.775** | **0.034** | 105 / 636 | **0.654** | 5 / 47 | 87 | **5.7%** | **0.075** | **42** |

Frame-level base rates are 0.81% for cross-validation and 1.35% for test. A detector that placed switches at random would be right about 0.8% of the time on the training students and about 1.3% on the test students. So the forest is roughly 3–5× better than chance, but still far from usable.

An earlier run used absolute pitch, roll and height instead of the `ConvTran/dataset.py` normalization, with labels covering 30 frames before and 29 after each switch. It scored about the same: cross-validation ROC-AUC 0.753 and test ROC-AUC 0.667.

### Gaze comparison

Same forest setup, but with `--gaze`: students 7, 8, 10 and 17 are dropped, motion clusters use the three gaze descriptors above, and the hand-built window features keep their gaze columns. The held-out test students are still 31 and 41 (47 switches). Training students now have 611 switches instead of 636. Outputs are under `Forest/output_gaze*`.

| Inputs | Cross-validation ROC-AUC | Cross-validation average precision | Cross-validation switches caught | Test ROC-AUC | Test switches caught | Test detections | Test precision | Test F1 | Test false alarms per hour |
|---|---|---|---|---|---|---|---|---|---|
| Hand-built window features (83) | 0.747 | 0.029 | 68 / 611 | 0.634 | 3 / 47 | 40 | 7.5% | 0.069 | 19 |
| 16 clusters (147 features) | 0.721 | 0.022 | 82 / 611 | 0.623 | 3 / 47 | 143 | 2.1% | 0.032 | 72 |
| 32 clusters (291 features) | 0.743 | 0.025 | 52 / 611 | 0.573 | 2 / 47 | 43 | 4.7% | 0.044 | 21 |
| 16 clusters + window features | 0.768 | 0.032 | 93 / 611 | 0.643 | 5 / 47 | 67 | 7.5% | 0.088 | 32 |

Against the headset-only table above:

- Gaze does **not** clearly improve detection. Cross-validation ROC-AUC is a little lower in every configuration.
- The combined gaze run is the least noisy of the gaze runs (test F1 0.088, 32 false alarms/hour) and edges the headset combined run on test F1, but still only catches 5 of 47 test switches.
- Gaze window features matter a little in the combined model (`gaze_speed_*` ranks in the mid teens of importance) but never dominate. Cluster-only gaze forests still lean on the same active/still 10 s clusters.

### What the forest relies on

- **Cluster-only forest:** the most important inputs are the 10 s cluster before and around each frame, especially the switch-prone cluster 13 and the still clusters 1 and 2, followed by whether the 10 s cluster changed.
- **Combined forest:** the 10 s variability features dominate, such as speed spread, angular-speed spread and mean speed.

Both say the same thing. Switches tend to happen during stretches of large head turns. But those stretches far more often occur without a switch, so motion alone cannot place a switch precisely.

## Possible next steps

1. Filter tracking glitches (cluster 15) before clustering and training.
2. Check before modelling whether any single motion measure changes consistently at switches, compared with random times. If none does, no model will find switches from headset motion alone.
3. Restrict the target to specific kinds of switch (particular code pairs), if some matter more.

## Reproducing

From `TimeSeriesModel/`:

```bash
venv/bin/python Forest/train_forest.py --features clusters --clusters 16 --out Forest/output_clusters16
venv/bin/python Forest/train_forest.py --features clusters --clusters 32 --out Forest/output_clusters32
venv/bin/python Forest/train_forest.py --features both --clusters 16 --out Forest/output_both16
venv/bin/python Forest/train_forest.py --features motion --out Forest/output

# Gaze cohort (drop students 7/8/10/17; gaze descriptors in clusters)
venv/bin/python Forest/train_forest.py --gaze --features clusters --clusters 16 --out Forest/output_gaze_clusters16
venv/bin/python Forest/train_forest.py --gaze --features clusters --clusters 32 --out Forest/output_gaze_clusters32
venv/bin/python Forest/train_forest.py --gaze --features both --clusters 16 --out Forest/output_gaze_both16
venv/bin/python Forest/train_forest.py --gaze --features motion --out Forest/output_gaze
```

- **Cache:** features and motion chunks are cached in `Forest/cache/`. The first run builds them in about a minute.
- **Run time:** each run takes 5–10 minutes on 32 cores.
- **Outputs in each run's folder:**
  - `metrics.json`
  - `forest.joblib`, containing the model, the cluster models, the feature list and the threshold
  - `feature_importances.csv`
  - `test_predictions.csv`
  - `pr_curve.png`
  - `cluster_profiles.csv` and `cluster_transitions.csv`, for runs that use clusters
- **Git:** the cache and output folders are git-ignored.

| File | Purpose |
|---|---|
| `features.py` | Resampling, per-frame window features, near-switch labels |
| `clusters.py` | Motion chunks, range-of-motion descriptors, k-means clusters, profiles and transitions |
| `train_forest.py` | Cross-validation, threshold choice, final model, switch-level evaluation, reports |
