"""Per-frame features and near-switch labels for the HeadsetFinal sessions.

Every session is resampled to 30 Hz with the same resampler the ConvTran
pipeline uses, so N frames is always N / 30 seconds. Each frame gets a row of
window statistics describing the head and gaze motion around it. The features
for a student are cached under Forest/cache so they are only built once.

Origin window features are expressed relative to the segment's median sitting
pose (median xyz, mean orientation), matching preprocessing.dataLoaders, so
standing or walking away from the desk remains visible as a large offset.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocessing.config import HEADSET_FINAL_DIR  # noqa: E402
from preprocessing.dataLoaders import (  # noqa: E402
    TARGET_HZ,
    _extract_subject_id,
    _parse_timestamps,
    _prepare_session_dataframe,
    resample_session_to_30hz,
)

GAZE_FEATURE_PREFIXES = ('gaze_',)

# Bump when the feature set changes so stale caches are rebuilt.
FEATURE_VERSION = 3
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

POSITION = ['headset_position_x', 'headset_position_y', 'headset_position_z']
ROTATION_WXYZ = ['headset_rotation_w', 'headset_rotation_x', 'headset_rotation_y', 'headset_rotation_z']
GAZE_DIRECTION = ['gaze_direction_x', 'gaze_direction_y', 'gaze_direction_z']

# Half widths of the centered windows: 1 s, 3 s and 10 s windows at 30 Hz.
CENTERED_HALF_WIDTHS = (15, 45, 150)
# Half widths for median-sitting-pose window features: 1 s and 3 s.
ORIGIN_HALF_WIDTHS = (15, 45)
# Frames on each side of the current frame for the before/after contrasts.
CONTRAST_WIDTHS = (30, 90)

CENTERED_STATS = {
    'mean': ['speed', 'ang_speed', 'gaze_speed', 'gaze_valid'],
    'std': ['speed', 'ang_speed', 'vel_y', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z'],
    'max': ['speed', 'ang_speed', 'gaze_speed'],
}
CONTRAST_SIGNALS = ['speed', 'ang_speed', 'gaze_speed', 'gaze_valid']

META_COLUMNS = ['subject', 'segment', 'frame', 'time_s', 'code']


def _seconds(frames):
    return '{:g}s'.format(frames / TARGET_HZ)


def _unit_quaternions(quats):
    """Normalize wxyz quaternions. A zero quaternion becomes the identity."""
    norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    good = np.isfinite(norm) & (norm > 1e-8)
    identity = np.zeros_like(quats)
    identity[..., 0] = 1.0
    return np.where(good, quats / np.where(good, norm, 1.0), identity)


def _quaternion_multiply(left, right):
    """Hamilton product of wxyz quaternions. Leading dimensions broadcast."""
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), axis=-1)


def relative_rotvec(start, quats):
    """Rotation of `quats` relative to `start`, as rotation vectors in radians.

    `inverse(start) * q`, so the vector is expressed in the reference pose's axes.
    """
    relative = _quaternion_multiply(start * np.array([1.0, -1.0, -1.0, -1.0]), quats)
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    vec_norm = np.linalg.norm(relative[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vec_norm, relative[..., :1])
    return relative[..., 1:] * (angle / np.maximum(vec_norm, 1e-12))


def _align_quaternion_signs(quats):
    out = np.asarray(quats, dtype=np.float64).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out


def _mean_unit_quaternion(quats):
    if len(quats) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    aligned = _align_quaternion_signs(quats)
    mean = aligned.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mean = mean / norm
    if mean[0] < 0.0:
        mean = -mean
    return mean


def _frame_signals(segment):
    """Per-frame motion signals for one gap-free 30 Hz segment.

    Each signal is a frame-to-frame change, so none of them depend on the
    absolute room pose.
    """
    n = len(segment)
    pos = segment[POSITION].to_numpy(dtype=np.float64)
    quat = _unit_quaternions(segment[ROTATION_WXYZ].to_numpy(dtype=np.float64))

    vel = np.zeros((n, 3))
    vel[1:] = np.diff(pos, axis=0) * TARGET_HZ
    ang_vel = np.zeros((n, 3))
    if n > 1:
        rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        ang_vel[1:] = (rot[:-1].inv() * rot[1:]).as_rotvec() * TARGET_HZ

    gaze = segment[GAZE_DIRECTION].to_numpy(dtype=np.float64)
    gaze_norm = np.linalg.norm(gaze, axis=1)
    # Missing gaze is stored as zeros, and the resampler blends those zeros
    # into neighbouring valid samples, so only near-unit vectors are trusted.
    gaze_valid = (gaze_norm > 0.5) & (gaze_norm < 1.5)
    gaze_unit = np.where(gaze_valid[:, None], gaze / np.maximum(gaze_norm, 1e-8)[:, None], 0.0)
    gaze_speed = np.zeros(n)
    if n > 1:
        dots = np.clip(np.sum(gaze_unit[1:] * gaze_unit[:-1], axis=1), -1.0, 1.0)
        both = gaze_valid[1:] & gaze_valid[:-1]
        gaze_speed[1:] = np.where(both, np.arccos(dots) * TARGET_HZ, 0.0)

    signals = pd.DataFrame({
        'speed': np.linalg.norm(vel, axis=1),
        'vel_y': vel[:, 1],
        'ang_speed': np.linalg.norm(ang_vel, axis=1),
        'ang_vel_x': ang_vel[:, 0],
        'ang_vel_y': ang_vel[:, 1],
        'ang_vel_z': ang_vel[:, 2],
        'gaze_valid': gaze_valid.astype(np.float64),
        'gaze_speed': gaze_speed,
    })
    return signals, pos, quat


def _origin_window_features(pos, quat, half, origin_pos, origin_quat, chunk=8192):
    """Window stats after centering the segment on its median sitting pose.

    The window covers frames t - half .. t + half (clipped to the segment).
    Position is relative to the segment median xyz; rotation is relative to the
    segment mean orientation — same convention as preprocessing.dataLoaders.
    """
    n = len(pos)
    offsets = np.arange(-half, half + 1)
    label = _seconds(2 * half)
    names = ['dist_mean', 'dist_max', 'rise_max', 'drop_max', 'height_mid',
             'angle_mean', 'angle_max', 'rot_x_range', 'rot_y_range', 'rot_z_range',
             'rot_x_mid', 'rot_y_mid']
    out = {name: np.empty(n) for name in names}
    ref_quat = np.broadcast_to(origin_quat.reshape(1, 4), (1, 4))
    for lo in range(0, n, chunk):
        rows = np.arange(lo, min(lo + chunk, n))
        idx = np.clip(rows[:, None] + offsets[None, :], 0, n - 1)
        rel_pos = pos[idx] - origin_pos.reshape(1, 1, 3)
        dist = np.linalg.norm(rel_pos, axis=-1)
        rotvec = relative_rotvec(ref_quat, quat[idx])
        angle = np.linalg.norm(rotvec, axis=-1)
        part = slice(lo, lo + len(rows))
        out['dist_mean'][part] = dist.mean(axis=1)
        out['dist_max'][part] = dist.max(axis=1)
        out['rise_max'][part] = rel_pos[..., 1].max(axis=1)
        out['drop_max'][part] = -rel_pos[..., 1].min(axis=1)
        out['height_mid'][part] = rel_pos[:, half, 1]
        out['angle_mean'][part] = angle.mean(axis=1)
        out['angle_max'][part] = angle.max(axis=1)
        for axis, name in enumerate('xyz'):
            out['rot_{}_range'.format(name)][part] = rotvec[..., axis].max(axis=1) - rotvec[..., axis].min(axis=1)
        out['rot_x_mid'][part] = rotvec[:, half, 0]
        out['rot_y_mid'][part] = rotvec[:, half, 1]
    return {'origin_{}_{}'.format(name, label): values for name, values in out.items()}


def segment_features(segment):
    """Feature table with one row per frame of a gap-free 30 Hz segment."""
    signals, pos, quat = _frame_signals(segment)
    n = len(signals)
    features = {}
    origin_pos = np.median(pos, axis=0)
    origin_quat = _mean_unit_quaternion(quat)

    for half in CENTERED_HALF_WIDTHS:
        window = signals.rolling(2 * half + 1, center=True, min_periods=1)
        label = _seconds(2 * half)
        for stat, columns in CENTERED_STATS.items():
            values = getattr(window[columns], stat)()
            if stat == 'std':
                values = values.fillna(0.0)
            for column in columns:
                features['{}_{}_{}'.format(column, stat, label)] = values[column].to_numpy()

    short, long_ = _seconds(2 * CENTERED_HALF_WIDTHS[0]), _seconds(2 * CENTERED_HALF_WIDTHS[-1])
    for column in ('speed', 'ang_speed'):
        features['{}_ratio_{}_{}'.format(column, short, long_)] = (
            features['{}_mean_{}'.format(column, short)]
            / (features['{}_mean_{}'.format(column, long_)] + 1e-6)
        )

    for half in ORIGIN_HALF_WIDTHS:
        features.update(_origin_window_features(pos, quat, half, origin_pos, origin_quat))

    contrast = signals[CONTRAST_SIGNALS]
    frames = np.arange(n)
    for width in CONTRAST_WIDTHS:
        after = contrast[::-1].rolling(width, min_periods=1).mean()[::-1]
        before = contrast.rolling(width, min_periods=1).mean().shift(1)
        before = before.fillna(after)
        label = _seconds(width)
        for column in CONTRAST_SIGNALS:
            features['{}_change_{}'.format(column, label)] = (after[column] - before[column]).to_numpy()

        # Pose at t + width relative to the pose at t - width.
        start = np.clip(frames - width, 0, n - 1)
        end = np.clip(frames + width, 0, n - 1)
        rotvec = relative_rotvec(quat[start], quat[end])
        features['net_displacement_{}'.format(label)] = np.linalg.norm(pos[end] - pos[start], axis=1)
        features['net_height_{}'.format(label)] = pos[end, 1] - pos[start, 1]
        features['net_rotation_{}'.format(label)] = np.linalg.norm(rotvec, axis=1)
        features['net_rot_x_{}'.format(label)] = rotvec[:, 0]
        features['net_rot_y_{}'.format(label)] = rotvec[:, 1]

    return pd.DataFrame(features).astype(np.float32)


def resampled_segments(path):
    """Yield (subject, segment id, start time in s, segment) for one student file.

    Each segment is a gap-free stretch on the 30 Hz grid. The start time is
    measured from the student's first frame.
    """
    subject = _extract_subject_id(path)
    raw = pd.read_csv(path, low_memory=False)
    raw = _prepare_session_dataframe(raw, path)
    data = resample_session_to_30hz(raw, path)
    data['auto_code'] = pd.to_numeric(data['auto_code'], errors='coerce').ffill().bfill()

    times = _parse_timestamps(data['timestamp'])
    t0 = times.iloc[0]
    for segment_id, segment in data.groupby('num_sessions', sort=True):
        start_s = (times.loc[segment.index[0]] - t0).total_seconds()
        yield subject, segment_id, start_s, segment.reset_index(drop=True)


def build_student(path):
    """Resample one student file and return its metadata plus feature rows."""
    parts = []
    for subject, segment_id, start_s, segment in resampled_segments(path):
        n = len(segment)
        meta = pd.DataFrame({
            'subject': np.full(n, subject, dtype=np.int32),
            'segment': np.full(n, segment_id, dtype=np.int32),
            'frame': np.arange(n, dtype=np.int32),
            'time_s': start_s + np.arange(n) / TARGET_HZ,
            'code': segment['auto_code'].to_numpy().astype(np.int16),
        })
        parts.append(pd.concat([meta, segment_features(segment)], axis=1))
    return pd.concat(parts, ignore_index=True)


def switch_times(time_s, codes):
    """Times of the first frame after each change of label code."""
    codes = np.asarray(codes)
    return np.asarray(time_s)[np.flatnonzero(codes[1:] != codes[:-1]) + 1]


def near_switch_labels(time_s, codes, window_frames):
    """1 for frames within `window_frames` of a code change on either side, else 0.

    A switch at frame s marks frames s - window_frames .. s + window_frames.
    Times are used instead of row indices so a switch is not stretched across
    a recording gap.
    """
    time_s = np.asarray(time_s)
    switches = switch_times(time_s, codes)
    if len(switches) == 0:
        return np.zeros(len(time_s), dtype=np.int8)
    width = window_frames / TARGET_HZ
    # Half a frame of slack absorbs millisecond rounding in the timestamps.
    slack = 0.5 / TARGET_HZ
    lo = np.searchsorted(switches, time_s - width - slack, side='left')
    hi = np.searchsorted(switches, time_s + width + slack, side='right')
    return (hi > lo).astype(np.int8)


def _cache_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, '{}_v{}.pkl'.format(name, FEATURE_VERSION))


def _build_and_cache(path):
    table = build_student(path)
    table.to_pickle(_cache_path(path))
    return path


def load_dataset(
    window_frames=30,
    rebuild=False,
    jobs=8,
    data_dir=HEADSET_FINAL_DIR,
    exclude_subjects=None,
    include_gaze_features=True,
):
    """All students' feature rows with a `label` column for near-switch frames."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError('No CSV files found under {}'.format(data_dir))
    excluded = set(exclude_subjects or ())
    if excluded:
        files = [path for path in files if _extract_subject_id(path) not in excluded]
    if not files:
        raise FileNotFoundError('No CSV files left under {} after exclusions {}'.format(
            data_dir, sorted(excluded)
        ))
    os.makedirs(CACHE_DIR, exist_ok=True)

    missing = [path for path in files if rebuild or not os.path.exists(_cache_path(path))]
    if missing:
        print('Building features for {} file(s)'.format(len(missing)))
        with Pool(processes=max(1, min(jobs, len(missing)))) as pool:
            for path in pool.imap_unordered(_build_and_cache, missing):
                print('  cached {}'.format(os.path.basename(path)))

    tables = []
    for path in files:
        table = pd.read_pickle(_cache_path(path))
        table['label'] = near_switch_labels(table['time_s'], table['code'], window_frames)
        tables.append(table)
    data = pd.concat(tables, ignore_index=True)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS + ['label']]
    if not include_gaze_features:
        feature_columns = [
            c for c in feature_columns
            if not any(c == prefix or c.startswith(prefix) for prefix in GAZE_FEATURE_PREFIXES)
        ]
    return data, feature_columns
