"""Export ConvTran predictions for every time point in the HeadsetFinal test set.

Uses the formatted test CSVs (students 31 and 41). Each row is one sample on
the 30 Hz grid built during preprocessing. The model sees a 240-step window
starting at that time point (8 seconds, same alignment as training) and
predicts whether exactly one label transition falls inside that window.

Output CSV columns:
    unique_id, timestep_index, timestamp, milliseconds,
    prediction, transition_prob, label
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from preprocessing.config import HEADSET_FINAL_DIR, change_data_path
from .config import apply_run_paths, convTranConfig, modelConfig
from .dataset import position_channel_index, rotation_channel_index
from .Models.model import model_factory
from .Models.utils import load_model


TEST_X_PATH = change_data_path('test', 'x')
TEST_Y_PATH = change_data_path('test', 'y')
MODEL_PATH = os.path.join(convTranConfig['save_dir'], 'AEF_model_last.pth')
OUTPUT_PATH = os.path.join(convTranConfig['pred_dir'], 'headset_final_test_predictions.csv')

BATCH_SIZE = 256


def _session_unique_id(subject, study_num=0, num_sessions=0):
    return '{}_{}_{}'.format(subject, study_num, num_sessions)


def _subject_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(?:student|subject)[_-]?(\d+)', basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _timestamp_columns(header):
    usecols = []
    for name in ('timestamp', 'Timestamp'):
        if name in header and name not in usecols:
            usecols.append(name)
    if 'milliseconds' in header:
        usecols.append('milliseconds')
    return usecols


def load_source_timestamps(unique_ids):
    """Map each test unique_id to original HeadsetFinal timestamp columns."""
    needed = set(unique_ids)
    lookup = {}

    for filename in glob.glob(os.path.join(HEADSET_FINAL_DIR, '**', '*.csv'), recursive=True):
        subject = _subject_from_filename(filename)
        if subject is None:
            continue
        uid = _session_unique_id(subject)
        if uid not in needed or uid in lookup:
            continue

        header = pd.read_csv(filename, nrows=0).columns.tolist()
        usecols = _timestamp_columns(header)
        if not usecols:
            print('No timestamp column in {}; skipping.'.format(filename))
            continue
        source = pd.read_csv(filename, usecols=usecols, low_memory=False)
        if 'timestamp' not in source.columns and 'Timestamp' in source.columns:
            source = source.rename(columns={'Timestamp': 'timestamp'})
        lookup[uid] = source.reset_index(drop=True)

    missing = needed - set(lookup)
    if missing:
        print('No HeadsetFinal timestamps for: {}'.format(sorted(missing)))
    return lookup


def window_labels(labels, starts, max_timestep):
    """Label is positive when exactly one transition falls inside the window."""
    length = len(labels)
    starts = np.asarray(starts)
    cumulative = np.concatenate([[0], np.cumsum(labels.astype(np.int32))])
    start_idx = np.clip(starts, 0, length)
    end_idx = np.clip(starts + max_timestep, 0, length)
    return (cumulative[end_idx] - cumulative[start_idx] == 1).astype(np.int8)


def iter_feature_batches(features, max_timestep, batch_size, position_index=None, rotation_index=None):
    """Yield (start_indices, windows) with windows shaped (batch, channels, time).

    Pose is already relative to each session's median sitting pose from
    preprocessing. Padded frames past the end of the session stay at zero.
    `position_index` / `rotation_index` are accepted for call-site compatibility.
    """
    del position_index, rotation_index
    n_steps = features.shape[0]
    offsets = torch.arange(max_timestep, device=features.device)
    for begin in range(0, n_steps, batch_size):
        starts = torch.arange(begin, min(begin + batch_size, n_steps), device=features.device)
        frame_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        valid = frame_idx < n_steps
        clipped = frame_idx.clamp(0, n_steps - 1)
        windows = features[clipped]
        windows = windows.masked_fill(~valid.unsqueeze(-1), 0)
        yield starts, windows.permute(0, 2, 1).contiguous()


def load_trained_model(device, config, model_path):
    config = dict(config)
    config['device'] = device
    config['Data_shape'] = list(config['Data_shape'])
    model = model_factory(config)
    model = load_model(model, model_path)
    model.to(device)
    model.eval()
    return model


def _timestamps_for_session(session_y, n_steps, unique_id, timestamp_lookup):
    """Prefer the 30 Hz timestamps stored with the labels; fall back to raw files."""
    if 'timestamp' in session_y.columns and len(session_y) == n_steps:
        timestamps = session_y['timestamp'].astype(str).to_numpy()
        if 'milliseconds' in session_y.columns:
            milliseconds = pd.to_numeric(session_y['milliseconds'], errors='coerce').to_numpy()
        else:
            milliseconds = np.full(n_steps, np.nan)
        return timestamps, milliseconds
    if timestamp_lookup is None:
        print('No resampled timestamps for {}; writing timestep index only.'.format(unique_id))
        return np.array([''] * n_steps, dtype=object), np.full(n_steps, np.nan)
    return _session_timestamps(unique_id, n_steps, timestamp_lookup)


def _session_timestamps(unique_id, n_steps, timestamp_lookup):
    source = timestamp_lookup.get(unique_id)
    if source is None or len(source) != n_steps:
        if source is None:
            print('No original timestamps for {}; writing timestep index only.'.format(unique_id))
        else:
            print(
                'Timestamp length mismatch for {} ({} vs {}); writing timestep index only.'.format(
                    unique_id, len(source), n_steps
                )
            )
        return np.array([''] * n_steps, dtype=object), np.full(n_steps, np.nan)

    timestamps = source['timestamp'].astype(str).to_numpy()
    if 'milliseconds' in source.columns:
        milliseconds = pd.to_numeric(source['milliseconds'], errors='coerce').to_numpy()
    else:
        milliseconds = np.full(n_steps, np.nan)
    return timestamps, milliseconds


def export_test_predictions(gaze=False):
    device = convTranConfig['device'] if torch.cuda.is_available() else 'cpu'
    max_timestep = modelConfig.max_timestep
    config = apply_run_paths(convTranConfig, gaze=gaze)
    test_x_path = change_data_path('test', 'x', gaze=gaze)
    test_y_path = change_data_path('test', 'y', gaze=gaze)
    model_path = os.path.join(config['save_dir'], config['last_checkpoint_name'])
    output_path = os.path.join(config['pred_dir'], config['prediction_name'])

    features_df = pd.read_csv(test_x_path)
    labels_df = pd.read_csv(test_y_path)
    features_df.fillna(0, inplace=True)
    labels_df['auto_code'] = labels_df['auto_code'].fillna(1)

    feature_cols = [col for col in features_df.columns if col != 'unique_id']
    position_index = position_channel_index(features_df.columns)
    rotation_index = rotation_channel_index(features_df.columns)
    print('Exporting HeadsetFinal test sessions: {}'.format(
        sorted(features_df['unique_id'].unique().tolist())
    ))
    if 'timestamp' in labels_df.columns:
        timestamp_lookup = None
    else:
        timestamp_lookup = load_source_timestamps(features_df['unique_id'].unique())

    n_features = len(feature_cols)
    config['Data_shape'] = [config['batch_size'], n_features, max_timestep]
    model = load_trained_model(device, config, model_path)
    session_frames = []

    grouped_features = features_df.groupby('unique_id', sort=False)
    grouped_labels = labels_df.groupby('unique_id', sort=False)

    with torch.inference_mode():
        for unique_id, session_x in grouped_features:
            session_y = grouped_labels.get_group(unique_id)
            features_np = session_x[feature_cols].to_numpy(dtype=np.float32)
            labels = session_y['auto_code'].to_numpy()
            n_steps = len(features_np)
            features = torch.from_numpy(features_np).to(device)
            timestamps, milliseconds = _timestamps_for_session(
                session_y, n_steps, unique_id, timestamp_lookup
            )

            pred_chunks = []
            prob_chunks = []
            label_chunks = []
            start_chunks = []
            n_batches = int(np.ceil(n_steps / float(BATCH_SIZE)))
            for starts, windows in tqdm(
                iter_feature_batches(
                    features, max_timestep, BATCH_SIZE, position_index, rotation_index
                ),
                total=n_batches,
                desc=str(unique_id),
            ):
                logits = model(windows)
                starts_np = starts.detach().cpu().numpy()
                pred_chunks.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int8))
                prob_chunks.append(softmax(logits, dim=1)[:, 1].cpu().numpy())
                label_chunks.append(window_labels(labels, starts_np, max_timestep))
                start_chunks.append(starts_np)

            starts_all = np.concatenate(start_chunks)
            session_frames.append(pd.DataFrame({
                'unique_id': unique_id,
                'timestep_index': starts_all,
                'timestamp': timestamps[starts_all],
                'milliseconds': milliseconds[starts_all],
                'prediction': np.concatenate(pred_chunks),
                'transition_prob': np.concatenate(prob_chunks),
                'label': np.concatenate(label_chunks),
            }))

    output = pd.concat(session_frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)

    print('Wrote {} rows to {}'.format(len(output), output_path))
    print('Prediction counts:\n{}'.format(output['prediction'].value_counts().to_string()))
    print('Label counts:\n{}'.format(output['label'].value_counts().to_string()))
    return output_path


if __name__ == '__main__':
    export_test_predictions()
