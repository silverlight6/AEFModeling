"""Zero-shot Telos-Small world model on headset transition windows.

Telos-Small (magilabsco/Telos-Small, 50.1M) encodes observed history and
predicts future patch latents in one shot, then reads those latents out as
21 quantiles. A window scores high when the median forecast misses the next
32 frames. The threshold is chosen on the training students to maximize
positive-class F1. Students 31 and 41 stay held out.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=1 venv/bin/python -m Telos.infer_transitions
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from Telos.load import cuda_device, load_telos, parameter_count
from Telos.metrics import best_f1_threshold, binary_scores, format_scores
from Telos.windows import load_split


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# Context is a multiple of the 32-step patch so the forecaster does not left-pad.
ORIGINS = (64, 96, 128, 160, 192)
HORIZON = 32


def _origin_scores(model, features, valid, batch_size):
    """Normalized median-forecast error at each origin. Shape (N, origins)."""
    device = next(model.parameters()).device
    median_index = model.config.median_index
    count = features.shape[0]
    scores = np.full((count, len(ORIGINS)), np.nan, dtype=np.float64)
    features = features.pin_memory()
    for origin_id, origin in enumerate(ORIGINS):
        eligible = torch.nonzero(valid >= origin + HORIZON, as_tuple=False).flatten()
        print("  origin {} horizon {}, {}/{} windows".format(
            origin, HORIZON, int(eligible.shape[0]), count
        ), flush=True)
        for start in range(0, int(eligible.shape[0]), batch_size):
            index = eligible[start:start + batch_size]
            context = features[index, :, :origin].to(device, non_blocking=True)
            future = features[index, :, origin:origin + HORIZON].to(device, non_blocking=True)
            quantiles = model.forecast(
                context,
                prediction_length=HORIZON,
                symmetric_averaging=True,
                clip_nonnegative=False,
            )
            median = quantiles[..., median_index]
            scale = context.std(dim=-1).clamp_min(1e-3)
            per_channel = (median - future).abs().mean(dim=-1) / scale
            scores[index.numpy(), origin_id] = per_channel.mean(dim=-1).float().cpu().numpy()
    return scores


def _reduce(origin_scores):
    """Peak and average surprise. Windows with no scored origin stay at 0."""
    present = ~np.isnan(origin_scores)
    all_missing = ~present.any(axis=1)
    filled = np.where(present, origin_scores, -np.inf)
    peak = filled.max(axis=1)
    peak[all_missing] = 0.0
    counts = present.sum(axis=1).clip(min=1)
    average = np.where(present, origin_scores, 0.0).sum(axis=1) / counts
    average[all_missing] = 0.0
    return {"max": peak, "mean": average}


def _choose_rule(labels, reductions):
    """Pick max or mean surprise by training F1, then keep that threshold."""
    chosen = None
    for name, scores in reductions.items():
        threshold = best_f1_threshold(labels, scores)
        predictions = (scores >= threshold).astype(int)
        metrics = binary_scores(labels, scores, predictions)
        record = {
            "name": name,
            "threshold": threshold,
            "train": metrics,
        }
        print("  train {} {}".format(name, format_scores("", metrics)), flush=True)
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        if chosen is None or key > chosen[0]:
            chosen = (key, record)
    return chosen[1]


def _write_predictions(path, split, scores, predictions, reductions):
    rows = []
    for row, (unique_id, window_start) in enumerate(split["hash_table"]):
        record = {
            "unique_id": unique_id,
            "window_start": int(window_start),
            "label": int(split["labels"][row]),
            "valid_length": int(split["valid"][row]),
            "score": float(scores[row]),
            "prediction": int(predictions[row]),
            "max_surprise": float(reductions["max"][row]),
            "mean_surprise": float(reductions["mean"][row]),
        }
        rows.append(record)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = cuda_device()
    print("Loading Telos-Small", flush=True)
    model = load_telos(device)
    print("Telos-Small parameters: {}".format(parameter_count(model)), flush=True)

    print("Loading windows", flush=True)
    train = load_split("train")
    test = load_split("test")
    print("Train windows: {} positive, {} negative".format(
        int((train["labels"] == 1).sum()), int((train["labels"] == 0).sum())
    ), flush=True)
    print("Test windows: {} positive, {} negative".format(
        int((test["labels"] == 1).sum()), int((test["labels"] == 0).sum())
    ), flush=True)

    print("Scoring training windows", flush=True)
    train_origins = _origin_scores(model, train["features"], train["valid"], batch_size=32)
    train_reductions = _reduce(train_origins)
    rule = _choose_rule(train["labels"].numpy(), train_reductions)
    print("Selected {} surprise, threshold {:.6f}".format(rule["name"], rule["threshold"]), flush=True)

    print("Scoring test windows", flush=True)
    test_origins = _origin_scores(model, test["features"], test["valid"], batch_size=32)
    test_reductions = _reduce(test_origins)
    test_scores = test_reductions[rule["name"]]
    test_predictions = (test_scores >= rule["threshold"]).astype(int)
    test_metrics = binary_scores(test["labels"].numpy(), test_scores, test_predictions)
    print("Test {}".format(format_scores("", test_metrics)), flush=True)

    prediction_path = os.path.join(OUTPUT_DIR, "test_window_scores.csv")
    _write_predictions(prediction_path, test, test_scores, test_predictions, test_reductions)
    summary = {
        "model": "Telos-Small",
        "checkpoint": "magilabsco/Telos-Small",
        "parameters": parameter_count(model),
        "gpu": torch.cuda.get_device_name(device),
        "task": "zero-shot forecast surprise on 8-second transition windows",
        "origins": list(ORIGINS),
        "horizon": HORIZON,
        "score": "per-channel context-normalized MAE of the 0.5 quantile, reduced over origins",
        "selected_reduction": rule["name"],
        "threshold": rule["threshold"],
        "selection": "highest positive-class F1 on the training students; precision then recall on ties",
        "train": rule["train"],
        "test": test_metrics,
        "train_windows": {
            "positive": int((train["labels"] == 1).sum()),
            "negative": int((train["labels"] == 0).sum()),
        },
        "test_windows": {
            "positive": int((test["labels"] == 1).sum()),
            "negative": int((test["labels"] == 0).sum()),
        },
        "predictions": prediction_path,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("Wrote {}".format(metrics_path), flush=True)


if __name__ == "__main__":
    main()
