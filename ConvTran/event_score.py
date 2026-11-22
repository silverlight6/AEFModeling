"""Event precision from 3-second headset motion peaks.

The threshold is chosen on the training sessions, then applied to each test
session on its own. A peak is one detection. Nearby peaks collapse into that
one detection, and it counts as a hit when a code change falls within a few
seconds.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from preprocessing.config import change_data_path

HZ = 30
MATCH_SECONDS = 5
MERGE_SECONDS = 5
MATCH_FRAMES = MATCH_SECONDS * HZ
MERGE_FRAMES = MERGE_SECONDS * HZ
MIN_TRAIN_RECALL = 0.25
POSITION_COL = 'max_headset_position_dist_3.0s'
ROTATION_COL = 'max_headset_rotation_dist_3.0s'


def _load_split(name, gaze=False):
    x_path = change_data_path(name, 'x', gaze=gaze)
    y_path = change_data_path(name, 'y', gaze=gaze)
    x = pd.read_csv(x_path, usecols=['unique_id', POSITION_COL, ROTATION_COL])
    y = pd.read_csv(y_path, usecols=['unique_id', 'auto_code'])
    y['auto_code'] = pd.to_numeric(y['auto_code'], errors='coerce').fillna(0).astype(np.int8)
    if len(x) != len(y) or not x['unique_id'].equals(y['unique_id']):
        raise RuntimeError('X/Y alignment failed for %s' % name)
    return x, y


def _score(position, rotation, position_scale, rotation_scale):
    pos = np.asarray(position, dtype=float) / position_scale
    rot = np.asarray(rotation, dtype=float) / rotation_scale
    return np.maximum(pos, rot)


def _peaks(score, height, merge_frames):
    """Local maxima at least `merge_frames` apart, each at or above `height`."""
    if len(score) < 3:
        return np.array([], dtype=int)
    indices, _ = find_peaks(score, height=height, distance=merge_frames)
    return indices


def _match(peaks, events, tolerance):
    """One-to-one match of peaks to change frames within `tolerance` frames."""
    unused = list(np.asarray(events, dtype=int))
    true_positives = 0
    for peak in peaks:
        best = None
        best_distance = tolerance + 1
        for event in unused:
            distance = abs(int(peak) - int(event))
            if distance < best_distance:
                best_distance = distance
                best = event
        if best is not None and best_distance <= tolerance:
            unused.remove(best)
            true_positives += 1
    false_positives = len(peaks) - true_positives
    false_negatives = len(unused)
    return true_positives, false_positives, false_negatives


def _session_arrays(x, y):
    sessions = []
    for uid, idx in x.groupby('unique_id', sort=False).groups.items():
        idx = np.asarray(list(idx))
        position = x.loc[idx, POSITION_COL].to_numpy(dtype=float)
        rotation = x.loc[idx, ROTATION_COL].to_numpy(dtype=float)
        events = np.flatnonzero(y.loc[idx, 'auto_code'].to_numpy() == 1)
        sessions.append((str(uid), position, rotation, events))
    return sessions


def _totals(sessions, score_scale, height):
    true_positives = false_positives = false_negatives = 0
    per_session = []
    position_scale, rotation_scale = score_scale
    for uid, position, rotation, events in sessions:
        score = _score(position, rotation, position_scale, rotation_scale)
        peaks = _peaks(score, height, MERGE_FRAMES)
        tp, fp, fn = _match(peaks, events, MATCH_FRAMES)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        per_session.append((uid, tp, fp, fn, len(peaks), len(events)))
    return true_positives, false_positives, false_negatives, per_session


def _rates(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    return precision, recall


def score_distance_peaks(gaze=False):
    """Print train-chosen event precision, then the same threshold on each test session."""
    train_x, train_y = _load_split('train', gaze=gaze)
    test_x, test_y = _load_split('test', gaze=gaze)
    position_scale = max(float(np.percentile(train_x[POSITION_COL], 95)), 1e-8)
    rotation_scale = max(float(np.percentile(train_x[ROTATION_COL], 95)), 1e-8)
    scale = (position_scale, rotation_scale)
    train_sessions = _session_arrays(train_x, train_y)
    test_sessions = _session_arrays(test_x, test_y)

    # Score units are "fraction of the training 95th percentile".
    heights = np.round(np.linspace(0.8, 4.0, 17), 3)
    best = None
    print('Peak match tolerance %ds, merge gap %ds' % (MATCH_SECONDS, MERGE_SECONDS))
    print('Train threshold sweep (precision at recall >= %.2f is preferred)' % MIN_TRAIN_RECALL)
    for height in heights:
        tp, fp, fn, _ = _totals(train_sessions, scale, height)
        precision, recall = _rates(tp, fp, fn)
        print('  height %.3f  precision %.3f  recall %.3f  tp %d fp %d fn %d' % (
            height, precision, recall, tp, fp, fn
        ))
        eligible = recall >= MIN_TRAIN_RECALL
        rank = (1 if eligible else 0, precision if eligible else 0.0, recall)
        if best is None or rank > best[0]:
            best = (rank, height, precision, recall)

    height = best[1]
    print('Chosen train height %.3f  precision %.3f  recall %.3f' % (height, best[2], best[3]))
    _, _, _, per_session = _totals(test_sessions, scale, height)
    print('Test sessions at that height')
    for uid, tp, fp, fn, n_peaks, n_events in per_session:
        precision, recall = _rates(tp, fp, fn)
        print('  %s  precision %.3f  recall %.3f  tp %d fp %d fn %d  peaks %d  changes %d' % (
            uid, precision, recall, tp, fp, fn, n_peaks, n_events
        ))
    return height


if __name__ == '__main__':
    score_distance_peaks()
