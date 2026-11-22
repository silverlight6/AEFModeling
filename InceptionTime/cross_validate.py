"""Leave-two-students-out rounds for InceptionTime, on the first GPU only.

Each round holds out a different pair of students as the test set and trains
on everyone else. Every round runs twice: once with class-balanced cross
entropy and once with unweighted cross entropy (the loss ConvTran uses). Each
run trains for the full epoch budget, 200 by default to match ConvTran, with
no early stopping.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=0 venv/bin/python -m InceptionTime.cross_validate

Round 1 is students 31 and 41, the ConvTran test pair. The other rounds use
disjoint pairs drawn with a fixed seed from students who have at least
`--min-positive-windows` transition windows, so every test set has positives.

A run whose `metrics.json` already exists is skipped, so an interrupted job
picks up where it stopped when rerun with the same arguments.
"""

import os

# Hide every GPU except the first before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from ConvTran.config import modelConfig
from ConvTran.dataset import AEFConvDataset
from ConvTran.Models.optimizers import get_optimizer
from preprocessing.config import FORMATTED_DIR, HEADSET_FINAL_TEST_SUBJECTS

from .model import InceptionTime, count_parameters
from .train import (
    SEED,
    _class_weights,
    _format_scores,
    _materialize,
    _metrics,
    _run_epoch,
    _selection_key,
    _set_seed,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_outputs")
LOSSES = ("balanced", "unweighted")
SUMMARY_METRICS = ("precision", "recall", "f1", "accuracy", "auroc", "average_precision")


def _subject(unique_id):
    return int(str(unique_id).split("_")[0])


class _GpuBatches:
    """Mini-batches drawn from tensors that already sit on the GPU."""

    def __init__(self, features, labels, ids, batch_size, shuffle=False, generator=None):
        self.features = features
        self.labels = labels
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        count = len(self.labels)
        if self.shuffle:
            order = torch.randperm(count, device=self.labels.device, generator=self.generator)
        else:
            order = torch.arange(count, device=self.labels.device)
        for start in range(0, count, self.batch_size):
            index = order[start:start + self.batch_size]
            # BatchNorm cannot train on a single window.
            if self.shuffle and len(index) < 2:
                continue
            yield self.features[index], self.labels[index], self.ids[index]


def load_all_windows():
    """Every 240-frame (8 s) window from the formatted train and test CSVs.

    Windows, labels, and pose centering come from `AEFConvDataset`, so they
    match what ConvTran trains on (exactly one transition => positive).
    """
    window_config = modelConfig()
    features = []
    labels = []
    index = []
    for split in ("train", "test"):
        dataset = AEFConvDataset(
            os.path.join(FORMATTED_DIR, "{}_change_data_x.csv".format(split)),
            os.path.join(FORMATTED_DIR, "{}_change_data_y.csv".format(split)),
            window_config,
            augment=False,
        )
        tensors = _materialize(dataset, split)
        features.append(tensors.tensors[0])
        labels.append(tensors.tensors[1])
        index.extend(dataset.hash_table)
    features = torch.cat(features)
    labels = torch.cat(labels)
    subjects = np.array([_subject(unique_id) for unique_id, _ in index])
    return features, labels, subjects, index


def subject_window_counts(labels, subjects):
    counts = defaultdict(lambda: [0, 0])
    for subject, label in zip(subjects.tolist(), labels.tolist()):
        counts[subject][0] += 1
        counts[subject][1] += int(label)
    return {subject: tuple(value) for subject, value in counts.items()}


def choose_test_pairs(counts, rounds, seed, min_positive_windows):
    """Round 1 is the ConvTran pair. Later rounds never reuse a student."""
    first_pair = tuple(sorted(HEADSET_FINAL_TEST_SUBJECTS))
    eligible = sorted(
        subject for subject, (_, positive) in counts.items()
        if positive >= min_positive_windows and subject not in first_pair
    )
    needed = 2 * (rounds - 1)
    if len(eligible) < needed:
        raise ValueError(
            "Only {} students have at least {} positive windows; {} rounds need {}".format(
                len(eligible), min_positive_windows, rounds, needed
            )
        )
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(eligible))
    pairs = [first_pair]
    for start in range(0, needed, 2):
        pairs.append(tuple(sorted(int(subject) for subject in shuffled[start:start + 2])))
    return pairs


def _round_dir(output_dir, round_number, pair):
    return os.path.join(output_dir, "round_{:02d}_s{}_s{}".format(round_number, pair[0], pair[1]))


def _prediction_frame(probabilities, targets, sample_ids, index):
    rows = []
    for sample_id, target, probability in zip(sample_ids, targets, probabilities):
        unique_id, window_start = index[int(sample_id)]
        rows.append({
            "unique_id": unique_id,
            "window_start": int(window_start),
            "label": int(target),
            "transition_prob": float(probability[1]),
            "prediction": int(probability.argmax()),
        })
    return pd.DataFrame(rows)


def _save_checkpoint(path, epoch, model, optimizer, extra):
    data = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    data.update(extra)
    torch.save(data, path)


def run_one(features, labels, subjects, index, pair, round_number, loss_name, run_dir,
            epochs, batch_size, lr, checkpoint_interval, device):
    seed = SEED + round_number
    _set_seed(seed)

    test_mask = np.isin(subjects, pair)
    train_rows = torch.from_numpy(np.flatnonzero(~test_mask)).to(device)
    test_rows = torch.from_numpy(np.flatnonzero(test_mask)).to(device)
    train_labels = labels[train_rows]
    test_labels = labels[test_rows]
    train_pos = int(train_labels.sum())
    test_pos = int(test_labels.sum())

    if loss_name == "balanced":
        weights = _class_weights(train_labels.cpu())
    else:
        weights = torch.ones(2)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    train_batches = _GpuBatches(
        features[train_rows], train_labels, train_rows, batch_size, shuffle=True, generator=generator
    )
    test_batches = _GpuBatches(features[test_rows], test_labels, test_rows, batch_size)

    model = InceptionTime(in_channels=features.shape[1], num_classes=2).to(device)
    optimizer = get_optimizer("AdamW")(model.parameters(), lr=lr, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), reduction="none")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_info = {
        "round": round_number,
        "test_subjects": list(pair),
        "loss": loss_name,
        "class_weights": weights.tolist(),
        "seed": seed,
    }
    print("Round {} test students {} loss {}: train {} windows ({} positive), test {} windows ({} positive), "
          "class weights {}".format(
              round_number, list(pair), loss_name, len(train_rows), train_pos,
              len(test_rows), test_pos, [round(w, 3) for w in weights.tolist()]
          ), flush=True)

    history = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_scores = None
    started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_scores, _, _, _ = _run_epoch(model, train_batches, criterion, device, optimizer)
        with torch.no_grad():
            test_scores, _, _, _ = _run_epoch(model, test_batches, criterion, device)
        history.append({
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "train": train_scores,
            "test": test_scores,
        })

        key = _selection_key(test_scores)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_scores = test_scores
            _save_checkpoint(os.path.join(checkpoint_dir, "best_f1.pth"), epoch, model, optimizer,
                             dict(run_info, test=test_scores))
        if checkpoint_interval and epoch % checkpoint_interval == 0 and epoch != epochs:
            _save_checkpoint(os.path.join(checkpoint_dir, "epoch_{}.pth".format(epoch)), epoch, model,
                             optimizer, dict(run_info, test=test_scores))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - started
            print("  epoch {:3d}/{} ({:.0f}s total) {}".format(
                epoch, epochs, elapsed, _format_scores("test", test_scores)
            ), flush=True)
            with open(os.path.join(run_dir, "history.json"), "w") as handle:
                json.dump(history, handle, indent=2)

    final_path = os.path.join(checkpoint_dir, "final.pth")
    _save_checkpoint(final_path, epochs, model, optimizer, dict(run_info, test=history[-1]["test"]))
    with open(os.path.join(run_dir, "history.json"), "w") as handle:
        json.dump(history, handle, indent=2)

    with torch.no_grad():
        final_scores, probabilities, targets, sample_ids = _run_epoch(model, test_batches, criterion, device)
    _prediction_frame(probabilities, targets, sample_ids, index).to_csv(
        os.path.join(run_dir, "test_predictions_final.csv"), index=False
    )
    best = torch.load(os.path.join(checkpoint_dir, "best_f1.pth"), map_location=device)
    model.load_state_dict(best["state_dict"])
    with torch.no_grad():
        _, probabilities, targets, sample_ids = _run_epoch(model, test_batches, criterion, device)
    _prediction_frame(probabilities, targets, sample_ids, index).to_csv(
        os.path.join(run_dir, "test_predictions_best_f1.csv"), index=False
    )

    metrics = dict(run_info)
    metrics.update({
        "model": "InceptionTime",
        "gpu": torch.cuda.get_device_name(device),
        "parameters": count_parameters(model),
        "epochs": epochs,
        "train_windows": {"positive": train_pos, "negative": len(train_rows) - train_pos},
        "test_windows": {"positive": test_pos, "negative": len(test_rows) - test_pos},
        "final_test": final_scores,
        "best_epoch": best_epoch,
        "best_test": best_scores,
        "seconds": time.time() - started,
        "checkpoints": {
            "final": final_path,
            "best_f1": os.path.join(checkpoint_dir, "best_f1.pth"),
        },
    })
    with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print("Round {} {} done in {:.0f} min. Final epoch {}. Best F1 epoch {} {}".format(
        round_number, loss_name, metrics["seconds"] / 60,
        _format_scores("", final_scores), best_epoch, _format_scores("", best_scores)
    ), flush=True)
    return metrics


def _pooled(rows, prefix):
    tp = sum(row[prefix + "_tp"] for row in rows)
    fp = sum(row[prefix + "_fp"] for row in rows)
    fn = sum(row[prefix + "_fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def write_summary(output_dir):
    """One row per finished run, plus per-loss averages and pooled counts."""
    rows = []
    for name in sorted(os.listdir(output_dir)):
        round_dir = os.path.join(output_dir, name)
        if not (name.startswith("round_") and os.path.isdir(round_dir)):
            continue
        for loss_name in LOSSES:
            path = os.path.join(round_dir, loss_name, "metrics.json")
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                metrics = json.load(handle)
            row = {
                "round": metrics["round"],
                "test_subjects": "{} {}".format(*metrics["test_subjects"]),
                "loss": loss_name,
                "test_positive_windows": metrics["test_windows"]["positive"],
                "test_windows": metrics["test_windows"]["positive"] + metrics["test_windows"]["negative"],
                "best_epoch": metrics["best_epoch"],
            }
            for prefix, scores in (("final", metrics["final_test"]), ("best", metrics["best_test"])):
                for key in SUMMARY_METRICS + ("tp", "fp", "fn", "tn"):
                    row["{}_{}".format(prefix, key)] = scores[key]
            rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["round", "loss"])
    frame.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    summary = {}
    for loss_name in LOSSES:
        loss_rows = [row for row in rows if row["loss"] == loss_name]
        if not loss_rows:
            continue
        entry = {"rounds_finished": len(loss_rows)}
        for prefix in ("final", "best"):
            stats = {}
            for key in SUMMARY_METRICS:
                values = [row["{}_{}".format(prefix, key)] for row in loss_rows
                          if row["{}_{}".format(prefix, key)] is not None]
                stats[key] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
                }
            stats["pooled"] = _pooled(loss_rows, prefix)
            entry[prefix] = stats
        summary[loss_name] = entry
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Leave-two-students-out rounds for InceptionTime")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--only-rounds", type=int, nargs="*",
                        help="Run just these round numbers (1-based); the pairs stay the same")
    parser.add_argument("--losses", nargs="*", default=list(LOSSES), choices=LOSSES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--min-positive-windows", type=int, default=10)
    parser.add_argument("--pair-seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training is pinned to GPU 0.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    print("Using GPU 0 only: {}".format(torch.cuda.get_device_name(device)), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    features, labels, subjects, index = load_all_windows()
    counts = subject_window_counts(labels, subjects)
    pairs = choose_test_pairs(counts, args.rounds, args.pair_seed, args.min_positive_windows)
    plan = [
        {
            "round": number,
            "test_subjects": list(pair),
            "test_windows": sum(counts[s][0] for s in pair),
            "test_positive_windows": sum(counts[s][1] for s in pair),
        }
        for number, pair in enumerate(pairs, start=1)
    ]
    with open(os.path.join(args.output_dir, "rounds.json"), "w") as handle:
        json.dump({
            "pair_seed": args.pair_seed,
            "min_positive_windows": args.min_positive_windows,
            "epochs": args.epochs,
            "rounds": plan,
        }, handle, indent=2)
    for entry in plan:
        print("Round {:2d}: test students {} ({} windows, {} positive)".format(
            entry["round"], entry["test_subjects"], entry["test_windows"], entry["test_positive_windows"]
        ), flush=True)

    features = features.to(device)
    labels = labels.to(device)
    for number, pair in enumerate(pairs, start=1):
        if args.only_rounds and number not in args.only_rounds:
            continue
        for loss_name in args.losses:
            run_dir = os.path.join(_round_dir(args.output_dir, number, pair), loss_name)
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                print("Round {} {} already finished, skipping".format(number, loss_name), flush=True)
                continue
            os.makedirs(run_dir, exist_ok=True)
            run_one(features, labels, subjects, index, pair, number, loss_name, run_dir,
                    args.epochs, args.batch_size, args.lr, args.checkpoint_interval, device)
            write_summary(args.output_dir)
    write_summary(args.output_dir)
    print("Wrote {}".format(os.path.join(args.output_dir, "summary.csv")), flush=True)


if __name__ == "__main__":
    main()
