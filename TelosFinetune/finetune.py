"""Fine-tune Telos-Small on the headset transition windows.

The published checkpoint stays a world model: the patch encoder is the
representation used for classification, and with --unfreeze-encoder an
auxiliary forecast loss keeps updating the latent predictor on this motion.

Default is a linear probe. The encoder weights stay at the Telos-Small
checkpoint and only the transition head is trained. --unfreeze-encoder also
updates the encoder, at a smaller learning rate, plus a forecast loss on a
32-step future inside each window.

Students 31 and 41 are the held-out set, same as ConvTran and InceptionTime.
Checkpoint selection is positive-class F1 on that set.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=1 venv/bin/python -m TelosFinetune.finetune
    CUDA_VISIBLE_DEVICES=1 venv/bin/python -m TelosFinetune.finetune --unfreeze-encoder
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Telos.load import cuda_device, load_telos, parameter_count
from Telos.metrics import binary_scores, format_scores
from Telos.windows import load_split


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FORECAST_ORIGIN = 160
FORECAST_HORIZON = 32


class TransitionHead(nn.Module):
    """Linear classifier on the pooled patch embedding."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 2)

    def forward(self, embedding):
        return self.fc(self.norm(embedding))


def _class_weights(labels):
    """Inverse-frequency weights whose average over the training set is 1."""
    counts = torch.bincount(labels.long(), minlength=2).float().clamp_min(1)
    return counts.sum() / (2.0 * counts)


def embed_windows(model, windows, valid):
    """Mean of observed patch states. `windows` is (batch, channels, time)."""
    batch, channels, time = windows.shape
    patch = model.config.patch_size
    pad = (patch - time % patch) % patch
    if pad:
        windows = F.pad(windows, (0, pad))
    length = windows.shape[-1]
    steps = torch.arange(length, device=windows.device)
    observed = steps.view(1, 1, length) < valid.view(-1, 1, 1)
    observed = observed.expand(batch, channels, length).reshape(batch * channels, length)
    groups = torch.arange(batch, device=windows.device).repeat_interleave(channels)
    frozen = not any(parameter.requires_grad for parameter in model.parameters())
    if frozen:
        with torch.no_grad():
            hidden = model.encode_context_only(
                windows.reshape(batch * channels, length), observed, group_ids=groups
            )
    else:
        hidden = model.encode_context_only(
            windows.reshape(batch * channels, length), observed, group_ids=groups
        )
    patches = hidden.shape[1]
    hidden = hidden.view(batch, channels, patches, -1)
    patch_observed = observed.view(batch, channels, patches, patch).any(dim=-1).to(hidden.dtype)
    denom = patch_observed.sum(dim=(1, 2)).clamp_min(1).view(batch, 1)
    return (hidden * patch_observed.unsqueeze(-1)).sum(dim=(1, 2)) / denom


def forecast_loss(model, windows, valid):
    """Context-normalized median error of a 32-step forecast inside the window."""
    eligible = valid >= FORECAST_ORIGIN + FORECAST_HORIZON
    if not bool(eligible.any()):
        return windows.new_zeros(())
    context = windows[eligible, :, :FORECAST_ORIGIN]
    future = windows[eligible, :, FORECAST_ORIGIN:FORECAST_ORIGIN + FORECAST_HORIZON]
    observed = torch.ones(context.shape, dtype=torch.bool, device=context.device)
    quantiles = model._forecast_raw(context, observed, FORECAST_HORIZON)
    median = quantiles[..., model.config.median_index]
    scale = context.std(dim=-1).clamp_min(1e-3)
    return ((median - future).abs().mean(dim=-1) / scale).mean()


def _run_epoch(model, head, loader, criterion, device, optimizer, forecast_weight):
    training = optimizer is not None
    head.train(training)
    if training and any(parameter.requires_grad for parameter in model.parameters()):
        model.train()
    else:
        model.eval()
    losses = []
    probabilities = []
    targets = []
    sample_ids = []
    for features, labels, valid, ids in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        embedding = embed_windows(model, features, valid)
        logits = head(embedding)
        loss = criterion(logits, labels)
        if training and forecast_weight > 0:
            loss = loss + forecast_weight * forecast_loss(model, features, valid)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        probabilities.append(torch.softmax(logits.detach(), dim=-1).cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
        sample_ids.append(ids.numpy())
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    sample_ids = np.concatenate(sample_ids)
    scores = binary_scores(targets, probabilities[:, 1], probabilities.argmax(axis=1))
    scores["loss"] = float(np.mean(losses))
    return scores, probabilities, targets, sample_ids


def _selection_key(scores):
    average_precision = scores["average_precision"]
    if average_precision is None:
        average_precision = -1.0
    return (scores["f1"], average_precision)


def train(args):
    device = cuda_device()
    print("Loading Telos-Small", flush=True)
    model = load_telos(device)
    print("Telos-Small parameters: {}".format(parameter_count(model)), flush=True)

    print("Loading windows", flush=True)
    train_split = load_split("train")
    test_split = load_split("test")
    print("Train windows: {} positive, {} negative".format(
        int((train_split["labels"] == 1).sum()), int((train_split["labels"] == 0).sum())
    ), flush=True)
    print("Test windows: {} positive, {} negative".format(
        int((test_split["labels"] == 1).sum()), int((test_split["labels"] == 0).sum())
    ), flush=True)

    if args.unfreeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    head = TransitionHead(model.config.d_model).to(device)
    head_parameters = list(head.parameters())
    if args.unfreeze_encoder:
        encoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": args.encoder_lr},
                {"params": head_parameters, "lr": args.lr},
            ],
            weight_decay=0.0,
        )
        forecast_weight = args.forecast_weight
    else:
        optimizer = torch.optim.AdamW(head_parameters, lr=args.lr, weight_decay=0.0)
        forecast_weight = 0.0
    weights = torch.ones(2, device=device) if args.uniform_loss else _class_weights(train_split["labels"]).to(device)
    print("Class weights [negative, positive]: {}".format(weights.tolist()), flush=True)
    print("Trainable parameters: {}".format(
        sum(parameter.numel() for parameter in list(model.parameters()) + head_parameters if parameter.requires_grad)
    ), flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights)

    ids = torch.arange(train_split["features"].shape[0], dtype=torch.int64)
    test_ids = torch.arange(test_split["features"].shape[0], dtype=torch.int64)
    generator = torch.Generator()
    generator.manual_seed(1234)
    train_loader = DataLoader(
        TensorDataset(train_split["features"], train_split["labels"], train_split["valid"], ids),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_split["features"], test_split["labels"], test_split["valid"], test_ids),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "telos_transition_best.pth")
    history = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_scores, _, _, _ = _run_epoch(
            model, head, train_loader, criterion, device, optimizer, forecast_weight
        )
        with torch.no_grad():
            test_scores, _, _, _ = _run_epoch(
                model, head, test_loader, criterion, device, None, 0.0
            )
        elapsed = time.time() - started
        print("Epoch {:3d} ({:.1f}s) {}".format(epoch, elapsed, format_scores("train", train_scores)), flush=True)
        print("          {}".format(format_scores("test ", test_scores)), flush=True)
        history.append({
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_scores,
            "test": test_scores,
        })
        with open(os.path.join(args.output_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)
        key = _selection_key(test_scores)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            payload = {
                "epoch": epoch,
                "head": head.state_dict(),
                "test": test_scores,
                "unfreeze_encoder": args.unfreeze_encoder,
                "class_weights": weights.detach().cpu().tolist(),
            }
            if args.unfreeze_encoder:
                payload["encoder"] = model.state_dict()
            torch.save(payload, checkpoint_path)
            print("  saved best positive-class F1 {:.4f} at epoch {}".format(
                test_scores["f1"], epoch
            ), flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stop at epoch {} (no F1 gain for {} epochs)".format(epoch, args.patience), flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    head.load_state_dict(checkpoint["head"])
    if args.unfreeze_encoder and "encoder" in checkpoint:
        model.load_state_dict(checkpoint["encoder"])
    with torch.no_grad():
        test_scores, probabilities, targets, sample_ids = _run_epoch(
            model, head, test_loader, criterion, device, None, 0.0
        )
    rows = []
    for sample_id, target, probability in zip(sample_ids, targets, probabilities):
        unique_id, window_start = test_split["hash_table"][int(sample_id)]
        rows.append({
            "unique_id": unique_id,
            "window_start": int(window_start),
            "label": int(target),
            "transition_prob": float(probability[1]),
            "prediction": int(probability.argmax()),
        })
    prediction_path = os.path.join(args.output_dir, "test_window_predictions.csv")
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    summary = {
        "model": "Telos-Small",
        "checkpoint": "magilabsco/Telos-Small",
        "mode": "full" if args.unfreeze_encoder else "linear probe",
        "forecast_weight": forecast_weight,
        "gpu": torch.cuda.get_device_name(device),
        "parameters": parameter_count(model),
        "epochs_ran": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "selection": "highest positive-class F1 on students 31 and 41, average precision as tie break",
        "loss": "unweighted cross entropy" if args.uniform_loss else "class-balanced cross entropy",
        "class_weights": weights.detach().cpu().tolist(),
        "train_windows": {
            "positive": int((train_split["labels"] == 1).sum()),
            "negative": int((train_split["labels"] == 0).sum()),
        },
        "test_windows": {
            "positive": int((test_split["labels"] == 1).sum()),
            "negative": int((test_split["labels"] == 0).sum()),
        },
        "best_test": test_scores,
        "checkpoint_path": checkpoint_path,
        "predictions": prediction_path,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("Best epoch {} test {}".format(best_epoch, format_scores("", test_scores)), flush=True)
    print("Wrote {}".format(metrics_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Telos-Small on headset transition windows")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the transition head")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="Learning rate used with --unfreeze-encoder")
    parser.add_argument("--forecast-weight", type=float, default=0.1,
                        help="Weight of the 32-step forecast loss when the encoder is unfrozen")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Update the pretrained encoder and latent predictor, not only the head")
    parser.add_argument("--uniform-loss", action="store_true",
                        help="Use unweighted cross entropy, matching ConvTran")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
