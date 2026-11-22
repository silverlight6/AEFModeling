"""Positive-class scores used by the other headset transition models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_scores(labels, positive_score, predictions):
    """Precision, recall, and F1 for the transition class, plus ranking metrics."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    positive_score = np.asarray(positive_score, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions).astype(int).reshape(-1)
    precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    scores = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if len(np.unique(labels)) > 1 and np.isfinite(positive_score).all():
        scores["auroc"] = float(roc_auc_score(labels, positive_score))
        scores["average_precision"] = float(average_precision_score(labels, positive_score))
    else:
        scores["auroc"] = None
        scores["average_precision"] = None
    return scores


def format_scores(name, scores):
    average_precision = scores["average_precision"]
    if average_precision is None:
        average_precision = float("nan")
    prefix = "{} ".format(name) if name else ""
    return (
        "{}acc {:.3f} prec {:.3f} rec {:.3f} f1 {:.3f} "
        "tp {} fp {} fn {} ap {:.3f}"
    ).format(
        prefix,
        scores["accuracy"],
        scores["precision"],
        scores["recall"],
        scores["f1"],
        scores["tp"],
        scores["fp"],
        scores["fn"],
        average_precision,
    )


def best_f1_threshold(labels, scores):
    """Threshold on `scores` (high means transition) with the best training F1.

    Ties go to the threshold with higher precision, so a flat score does not
    collapse to flagging every window.
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    grid = np.concatenate([[scores.min() - 1.0], grid, [scores.max() + 1.0]])
    best_key = None
    best_threshold = float(grid[-1])
    for threshold in grid:
        predictions = (scores >= threshold).astype(int)
        precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
        recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
        key = (f1, precision, recall)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold
