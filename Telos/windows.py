"""Non-overlapping headset windows, same split and label rule as ConvTran."""

import os
import time

import numpy as np
import torch

from ConvTran.config import modelConfig
from ConvTran.dataset import AEFConvDataset, window_transition_label
from preprocessing.config import FORMATTED_DIR


def load_split(name):
    """Materialize one split.

    Returns features (N, channels, 240), labels, valid frame counts, and the
    (unique_id, window_start) pairs those rows came from. Augmentation is off:
    these rows are the stored pose, not a jittered training view.
    """
    dataset = AEFConvDataset(
        os.path.join(FORMATTED_DIR, "{}_change_data_x.csv".format(name)),
        os.path.join(FORMATTED_DIR, "{}_change_data_y.csv".format(name)),
        modelConfig(),
        augment=False,
    )
    started = time.time()
    columns = {}
    features = []
    labels = []
    valid_lengths = []
    length = dataset.max_timestep
    total = len(dataset.hash_table)
    for sample_id, (unique_id, start) in enumerate(dataset.hash_table):
        if unique_id not in columns:
            frame = dataset.student_dataset.get_group(unique_id).drop(columns="unique_id")
            values = frame.to_numpy(dtype=np.float32, copy=True)
            label_values = dataset.student_labels.get_group(unique_id)["auto_code"].to_numpy()
            columns[unique_id] = (values, label_values)
        values, label_values = columns[unique_id]
        end = min(len(values), start + length)
        valid = max(0, end - start) if start < len(values) else 0
        window = values[start:start + length]
        if valid < length:
            pad = np.zeros((length - valid, values.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0) if valid else pad
        tensor = torch.tensor(np.ascontiguousarray(window), dtype=torch.float32)
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, 0)
        features.append(tensor.transpose(0, 1).contiguous())
        if start < len(label_values):
            labels.append(window_transition_label(label_values[start:end]))
        else:
            labels.append(0)
        valid_lengths.append(valid)
        if sample_id and sample_id % 2000 == 0:
            print("  {}: {}/{} windows".format(name, sample_id, total), flush=True)
    print("  {}: {} windows in {:.1f}s, shape {}".format(
        name, total, time.time() - started, (total, features[0].shape[0], features[0].shape[1]) if features else (0,)
    ), flush=True)
    return {
        "features": torch.stack(features) if features else torch.zeros(0, 0, length),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "valid": torch.tensor(valid_lengths, dtype=torch.int64),
        "hash_table": list(dataset.hash_table),
    }
