"""Train InceptionTime on the same transition windows as ConvTran.

Run from the repository root, on the first GPU only:

    CUDA_VISIBLE_DEVICES=0 venv/bin/python -m InceptionTime.train

Windows, pose normalization, and the student split come from ConvTran's
dataset. An 8-second window is positive when exactly one label transition
falls inside it (same rule as TercVJepa / AEFConvDataset). Checkpoint
selection uses positive-class F1 on students 31 and 41, which is the same
held-out set ConvTran validates on.
"""

import os

# Hide every GPU except the first before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from ConvTran.config import modelConfig
from ConvTran.dataset import (
    AEFConvDataset,
    window_transition_label,
)
from ConvTran.Models.optimizers import get_optimizer
from preprocessing.config import FORMATTED_DIR

from .model import InceptionTime, count_parameters


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 1234


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _window_labels(dataset):
    """Positive-window count using the same rule as AEFConvDataset."""
    by_student = {}
    positive = 0
    for unique_id, start in dataset.hash_table:
        if unique_id not in by_student:
            by_student[unique_id] = dataset.student_labels.get_group(unique_id)["auto_code"].to_numpy()
        values = by_student[unique_id]
        end = min(len(values), start + dataset.max_timestep)
        if start < len(values) and window_transition_label(values[start:end]):
            positive += 1
    total = len(dataset.hash_table)
    return positive, total - positive


def _window_tensor(values, start, length):
    """One window, matching AEFConvDataset.__getitem__ (pose already median-centered)."""
    end = start + length
    window = values[start:end]
    valid = len(window)
    if valid < length:
        pad = np.zeros((length - valid, values.shape[1]), dtype=np.float32)
        window = np.concatenate([window, pad], axis=0) if valid else pad
    tensor = torch.tensor(np.ascontiguousarray(window), dtype=torch.float32)
    if torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, 0)
    return tensor.transpose(0, 1).contiguous()


def _materialize(dataset, name):
    """Copy windows into tensors once. Later epochs then skip the CSV path."""
    columns = {}
    started = time.time()
    features = []
    labels = []
    ids = []
    total = len(dataset.hash_table)
    for sample_id, (unique_id, start) in enumerate(dataset.hash_table):
        if unique_id not in columns:
            frame = dataset.student_dataset.get_group(unique_id).drop(columns="unique_id")
            values = frame.to_numpy(dtype=np.float32, copy=True)
            label_values = dataset.student_labels.get_group(unique_id)["auto_code"].to_numpy()
            columns[unique_id] = (values, label_values)
        values, label_values = columns[unique_id]
        features.append(_window_tensor(values, start, dataset.max_timestep))
        end = min(len(label_values), start + dataset.max_timestep)
        if start >= len(label_values):
            labels.append(0)
        else:
            labels.append(window_transition_label(label_values[start:end]))
        ids.append(sample_id)
        if sample_id and sample_id % 2000 == 0:
            print("  {}: {}/{} windows".format(name, sample_id, total), flush=True)
    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.int64)
    ids = torch.tensor(ids, dtype=torch.int64)
    print("  {}: {} windows in {:.1f}s, shape {}".format(
        name, total, time.time() - started, tuple(features.shape)
    ), flush=True)
    return TensorDataset(features, labels, ids)


def _class_weights(labels):
    """Inverse-frequency weights whose average over the training set is 1."""
    counts = torch.bincount(labels.long(), minlength=2).float().clamp_min(1)
    return counts.sum() / (2.0 * counts)


def _metrics(probabilities, targets):
    targets = np.asarray(targets).astype(int).reshape(-1)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    positive_prob = probabilities[:, 1]
    precision = precision_score(targets, predictions, pos_label=1, zero_division=0)
    recall = recall_score(targets, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(targets, predictions, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    scores = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if len(np.unique(targets)) > 1:
        scores["auroc"] = float(roc_auc_score(targets, positive_prob))
        scores["average_precision"] = float(average_precision_score(targets, positive_prob))
    else:
        scores["auroc"] = None
        scores["average_precision"] = None
    return scores


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    probabilities = []
    targets = []
    sample_ids = []
    for features, batch_targets, batch_ids in loader:
        features = features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, batch_targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
        total_loss += float(loss.sum().detach())
        total += len(batch_targets)
        probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())
        sample_ids.append(batch_ids.cpu().numpy())
    scores = _metrics(np.concatenate(probabilities), np.concatenate(targets))
    scores["loss"] = total_loss / max(total, 1)
    return scores, np.concatenate(probabilities), np.concatenate(targets), np.concatenate(sample_ids)


def _format_scores(prefix, scores):
    return (
        "{} loss {:.4f} acc {:.3f} prec {:.3f} rec {:.3f} f1 {:.3f} "
        "auroc {:.3f} ap {:.3f} tp {} fp {} fn {}"
    ).format(
        prefix,
        scores["loss"],
        scores["accuracy"],
        scores["precision"],
        scores["recall"],
        scores["f1"],
        scores["auroc"] if scores["auroc"] is not None else float("nan"),
        scores["average_precision"] if scores["average_precision"] is not None else float("nan"),
        scores["tp"],
        scores["fp"],
        scores["fn"],
    )


def _selection_key(scores):
    average_precision = scores["average_precision"]
    if average_precision is None:
        average_precision = -1.0
    return (scores["f1"], average_precision)


def train(epochs, patience, batch_size, num_workers, lr, output_dir, balanced):
    _set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training is pinned to GPU 0.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    print("Using GPU 0 only: {}".format(torch.cuda.get_device_name(device)), flush=True)

    window_config = modelConfig()
    train_dataset = AEFConvDataset(
        os.path.join(FORMATTED_DIR, "train_change_data_x.csv"),
        os.path.join(FORMATTED_DIR, "train_change_data_y.csv"),
        window_config,
        augment=True,
    )
    test_dataset = AEFConvDataset(
        os.path.join(FORMATTED_DIR, "test_change_data_x.csv"),
        os.path.join(FORMATTED_DIR, "test_change_data_y.csv"),
        window_config,
        augment=True,
    )
    train_pos, train_neg = _window_labels(train_dataset)
    test_pos, test_neg = _window_labels(test_dataset)
    print("Train windows: {} positive, {} negative".format(train_pos, train_neg), flush=True)
    print("Test windows: {} positive, {} negative".format(test_pos, test_neg), flush=True)

    print("Loading windows into memory", flush=True)
    train_tensors = _materialize(train_dataset, "train")
    test_tensors = _materialize(test_dataset, "test")
    if balanced:
        weights = _class_weights(train_tensors.tensors[1])
    else:
        weights = torch.ones(2)
    print("Class weights [negative, positive]: {}".format(weights.tolist()), flush=True)

    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_tensors, shuffle=True, generator=generator, **loader_kwargs)
    test_loader = DataLoader(test_tensors, shuffle=False, **loader_kwargs)

    model = InceptionTime(
        in_channels=train_tensors.tensors[0].shape[1],
        num_classes=2,
    ).to(device)
    print("InceptionTime parameters: {}".format(count_parameters(model)), flush=True)

    optimizer = get_optimizer("AdamW")(model.parameters(), lr=lr, weight_decay=0)
    # Targets are int64 class indices from _materialize / AEFConvDataset.
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), reduction="none")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "inception_best.pth")
    history = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_scores, _, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            test_scores, _, _, _ = _run_epoch(model, test_loader, criterion, device)
        elapsed = time.time() - started
        print("Epoch {:3d} ({:.1f}s) {}".format(epoch, elapsed, _format_scores("train", train_scores)), flush=True)
        print("          {}".format(_format_scores("test ", test_scores)), flush=True)
        history.append({"epoch": epoch, "seconds": elapsed, "train": train_scores, "test": test_scores})
        with open(os.path.join(output_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)

        key = _selection_key(test_scores)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test": test_scores,
                "class_weights": weights.tolist(),
            }, checkpoint_path)
            print("  saved best positive-class F1 {:.4f} at epoch {}".format(
                test_scores["f1"], epoch
            ), flush=True)
        else:
            stale += 1
            if stale >= patience:
                print("Early stop at epoch {} (no F1 gain for {} epochs)".format(epoch, patience), flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        test_scores, probabilities, targets, sample_ids = _run_epoch(
            model, test_loader, criterion, device
        )
    rows = []
    for sample_id, target, probability in zip(sample_ids, targets, probabilities):
        unique_id, window_start = test_dataset.hash_table[int(sample_id)]
        rows.append({
            "unique_id": unique_id,
            "window_start": int(window_start),
            "label": int(target),
            "transition_prob": float(probability[1]),
            "prediction": int(probability.argmax()),
        })
    prediction_path = os.path.join(output_dir, "test_window_predictions.csv")
    pd.DataFrame(rows).to_csv(prediction_path, index=False)

    summary = {
        "model": "InceptionTime",
        "gpu": torch.cuda.get_device_name(device),
        "parameters": count_parameters(model),
        "epochs_ran": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "selection": "highest positive-class F1 on students 31 and 41, average precision as tie break",
        "loss": "class-balanced cross entropy" if balanced else "unweighted cross entropy",
        "class_weights": weights.tolist(),
        "train_windows": {"positive": train_pos, "negative": train_neg},
        "test_windows": {"positive": test_pos, "negative": test_neg},
        "best_test": test_scores,
        "checkpoint": checkpoint_path,
        "predictions": prediction_path,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("Best epoch {} test {}".format(best_epoch, _format_scores("", test_scores)), flush=True)
    print("Wrote {}".format(metrics_path), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train InceptionTime on headset transition windows")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--uniform-loss", action="store_true",
                        help="Use unweighted cross entropy, matching ConvTran")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(
        args.epochs, args.patience, args.batch_size, args.num_workers, args.lr,
        args.output_dir, balanced=not args.uniform_loss,
    )


if __name__ == "__main__":
    main()
