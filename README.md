# ConvTran

ConvTran model for headset time series. Most of the model code comes from [ConvTran](https://github.com/Navidfoumani/ConvTran).

`python main.py` runs the full pipeline in one process:

1. Preprocess HeadsetFinal sessions into train and test change-point CSVs.
2. Train ConvTran on those CSVs.
3. Export per-timestep test predictions.

Raw sessions are read from `../Data/HeadsetFinal`. Formatted CSVs are written to `../Data/FormattedData`. Test subjects are 31 and 41. These are set in `preprocessing/config.py`.

## Layout

```
main.py                  Full pipeline: preprocess, train, export
preprocessing/
    config.py            Data paths and the train/test subject split
    dataLoaders.py       Resampling to 30 Hz, labels, motion features, formatted CSVs
    util.py              Train/test split
ConvTran/
    config.py            Model and training settings (convTranConfig)
    dataset.py           AEFConvDataset and pose normalization
    train.py             Builds the model and data loaders and runs training
    trainer.py           Training loop, validation and checkpointing
    event_score.py       Motion-peak baseline for event precision
    export_test_predictions.py   Per-timestep predictions on the test set
    Models/              ConvTran network, attention, positional encodings, losses
    outputs/             Checkpoints, prediction CSVs and TensorBoard logs (git-ignored)
InceptionTime/           Multi-scale residual CNN baseline for the same windows
Forest/                  Random forest baseline, see Forest/RESULTS.md
archive/                 Old checkpoints from earlier models (git-ignored)
```

## InceptionTime

`InceptionTime` is a separate classifier for the same task: an 8-second (240-frame at 30 Hz) window is positive when exactly one label transition falls inside it. It is a residual CNN with three kernel sizes in parallel, and it does not use attention. Training reads the formatted CSVs, keeps students 31 and 41 as the test set, and saves the checkpoint with the best positive-class F1.

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python -m InceptionTime.train
CUDA_VISIBLE_DEVICES=0 venv/bin/python -m InceptionTime.train --uniform-loss --output-dir InceptionTime/outputs_uniform
```

The first command uses class-balanced cross entropy and writes `InceptionTime/outputs/`. The second uses the same unweighted cross entropy as ConvTran. Both runs pin training to GPU 0. Each folder gets `metrics.json`, `history.json`, `checkpoints/inception_best.pth`, and `test_window_predictions.csv`.

On the held-out windows (40 of 410 contain a transition), positive-class F1 at the best checkpoint was:

| Model | Loss | Precision | Recall | F1 | Caught |
|---|---|---|---|---|---|
| ConvTran, best logged epoch | unweighted | 0.348 | 0.200 | 0.254 | 8 / 40 |
| InceptionTime | unweighted | 0.200 | 0.100 | 0.133 | 4 / 40 |
| InceptionTime | balanced | 0.178 | 0.450 | 0.255 | 18 / 40 |

### Rounds with different test students

`InceptionTime/cross_validate.py` repeats training with ten different test pairs. Round 1 is students 31 and 41; the other nine pairs are disjoint and drawn with a fixed seed from students with at least 10 transition windows. Each round trains twice, once with balanced loss and once with unweighted loss, for 200 epochs with no early stopping.

```bash
CUDA_VISIBLE_DEVICES=0 nohup venv/bin/python -u -m InceptionTime.cross_validate > InceptionTime/cv_outputs/run.log 2>&1 &
```

Each run writes `InceptionTime/cv_outputs/round_XX_sA_sB/<loss>/` with `checkpoints/final.pth` (epoch 200), `checkpoints/best_f1.pth`, `checkpoints/epoch_{50,100,150}.pth`, `history.json`, `metrics.json` and test predictions. `rounds.json` lists the pairs, and `summary.csv` and `summary.json` collect every finished run. A finished run is skipped on restart, so rerunning the same command resumes an interrupted job. One run takes about 53 minutes on the K80, so all 20 take about 18 hours.

ConvTran’s row is the best positive-class F1 in the 26 Oct 2026 TensorBoard run. With the same loss, ConvTran scores higher. Balanced InceptionTime matches that F1 and flags more transition windows, with more false positives.

## Outputs

Everything ConvTran writes goes under `ConvTran/outputs/`:

- `checkpoints/`: `AEF_model_last.pth` (best positive-class F1), `model_best.pth` (best `key_metric`) and periodic epoch checkpoints
- `predictions/headset_final_test_predictions.csv`
- `tensorboard/`: view with `tensorboard --logdir ConvTran/outputs/tensorboard`

## Running single steps

Run modules from the repository root so the package imports resolve:

```bash
python -m ConvTran.event_score
python -m ConvTran.export_test_predictions
```
