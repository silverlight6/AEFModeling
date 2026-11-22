import pandas as pd
import glob
import os
import re
import numpy as np
from numba import njit, prange
from scipy.spatial.transform import Rotation, Slerp
from preprocessing.util import train_test_split
from preprocessing.config import (
    FORMATTED_DIR,
    GAZE_EXCLUDED_SUBJECTS,
    HEADSET_FINAL_DIR,
    HEADSET_FINAL_TEST_SUBJECTS,
    change_data_path,
)


DROP_LIST_1 = [
    'timestamp',
    'AttentionStatus',
    'Unnamed: 0',
    'Unnamed: 15',
    'milliseconds',
    'label',
    'gaze_origin_x',
    'gaze_origin_y',
    'gaze_origin_z',
    'gaze_direction_x',
    'gaze_direction_y',
    'gaze_direction_z',
]

DROP_LIST_2 = [
    'auto_code',
]

FEATURES = [
    'headset_position_x',
    'headset_position_y',
    'headset_position_z',
    'headset_rotation_w',
    'headset_rotation_x',
    'headset_rotation_y',
    'headset_rotation_z',
]

# Distance windows and the 240-step (8 s) model window count frames as if every
# session was recorded at 30 Hz. HeadsetFinal is a mix of ~30 Hz and ~60 Hz.
TARGET_HZ = 30
# Gaps wider than this are dropouts. Each side is resampled on its own so the
# hole is not filled with invented samples, and a later window cannot step across it.
MAX_GAP_S = 1.0
TIMESTAMP_FORMAT = '%m/%d/%Y %I:%M:%S.%f %p'

GAZE_COLUMNS = [
    'gaze_origin_x',
    'gaze_origin_y',
    'gaze_origin_z',
    'gaze_direction_x',
    'gaze_direction_y',
    'gaze_direction_z',
]
GAZE_ORIGIN_COLUMNS = GAZE_COLUMNS[:3]
GAZE_DIRECTION_COLUMNS = GAZE_COLUMNS[3:]
HEADSET_DISTANCE_SIGNALS = ('headset_position', 'headset_rotation')
GAZE_DISTANCE_SIGNALS = ('gaze_origin', 'gaze_direction')


def _max_distance_columns(signal_names):
    """Column names for the 1s, 3s, and 10s max-distance channels, in that order."""
    columns = []
    for frames in (TARGET_HZ, TARGET_HZ * 3, TARGET_HZ * 10):
        for name in signal_names:
            columns.append('max_{}_dist_{}s'.format(name, frames / TARGET_HZ))
    return columns


# ConvTran expects 13 headset channels: 7 pose + 6 max-distance features.
MODEL_INPUT_COLUMNS = FEATURES + _max_distance_columns(HEADSET_DISTANCE_SIGNALS)
# Gaze dataset adds the 6 gaze channels and the same 1s/3s/10s distances for
# gaze origin and gaze direction. 25 channels in total.
GAZE_MODEL_INPUT_COLUMNS = (
    FEATURES
    + GAZE_COLUMNS
    + _max_distance_columns(HEADSET_DISTANCE_SIGNALS)
    + _max_distance_columns(GAZE_DISTANCE_SIGNALS)
)
ROTATION_XYZW = [
    'headset_rotation_x',
    'headset_rotation_y',
    'headset_rotation_z',
    'headset_rotation_w',
]
POSITION_COLUMNS = [
    'headset_position_x',
    'headset_position_y',
    'headset_position_z',
]
ROTATION_WXYZ = [
    'headset_rotation_w',
    'headset_rotation_x',
    'headset_rotation_y',
    'headset_rotation_z',
]


def _extract_subject_id(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(?:student|subject)[_-]?(\d+)', basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _prepare_session_dataframe(df, filename):
    if 'subject' not in df.columns:
        subject = _extract_subject_id(filename)
        if subject is None:
            raise ValueError('Could not infer subject id from {}'.format(filename))
        df['subject'] = subject
        df['study_num'] = 0
        df['num_sessions'] = 0
    if 'auto_code' not in df.columns:
        if 'label' not in df.columns:
            raise ValueError('No auto_code or label column in {}'.format(filename))
        df['auto_code'] = df['label']
    # Dated Sensor files stored labels in full_code instead of auto_code.
    basename = os.path.basename(filename)
    if basename.startswith('20') and 'full_code' in df.columns:
        df[['auto_code', 'full_code']] = df[['full_code', 'auto_code']]
    return df


def _parse_timestamps(series):
    parsed = pd.to_datetime(series, format=TIMESTAMP_FORMAT, errors='coerce')
    if parsed.notna().mean() < 0.5:
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed


def _format_timestamps(times):
    formatted = pd.Series(pd.DatetimeIndex(times)).dt.strftime('%m/%d/%Y %I:%M:%S.%f %p')
    return formatted.str.replace(r'(\.\d{3})\d+', r'\1', regex=True)


def _numeric_source(series):
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    if not np.isnan(values).any():
        return values
    filled = pd.Series(values).interpolate(limit_direction='both').ffill().bfill()
    return filled.fillna(0).to_numpy(dtype=np.float64)


def _resample_rotation(segment, src_s, tgt_s):
    quat = segment[ROTATION_XYZW].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    norms = np.linalg.norm(quat, axis=1)
    bad = ~np.isfinite(norms) | (norms < 1e-8)
    if bad.any():
        quat[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        norms = np.linalg.norm(quat, axis=1)
    quat = quat / norms[:, None]
    # Keep each step on the same hemisphere so slerp does not take the long way.
    dots = np.sum(quat[1:] * quat[:-1], axis=1)
    sign = np.ones(len(quat))
    sign[1:] = np.where(dots < 0.0, -1.0, 1.0)
    quat = quat * np.cumprod(sign)[:, None]
    slerped = Slerp(src_s, Rotation.from_quat(quat))(tgt_s).as_quat()
    return slerped


def _resample_segment(segment, target_hz):
    """Place one segment on an exact `target_hz` grid. Pose is interpolated; labels are held."""
    segment = segment.reset_index(drop=True)
    t0 = segment['_time'].iloc[0]
    src_s = (segment['_time'] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    keep = np.concatenate([[True], np.diff(src_s) > 0])
    if not keep.all():
        segment = segment.loc[keep].reset_index(drop=True)
        src_s = src_s[keep]
    if len(src_s) < 2 or src_s[-1] < 1.0 / target_hz:
        return None

    n_out = int(np.floor(src_s[-1] * target_hz + 1e-9)) + 1
    tgt_s = np.arange(n_out, dtype=np.float64) / float(target_hz)
    tgt_s = tgt_s[tgt_s <= src_s[-1] + 1e-12]
    tgt_s[-1] = min(float(tgt_s[-1]), float(src_s[-1]))

    out = pd.DataFrame()
    for column in FEATURES + GAZE_COLUMNS:
        if column in ROTATION_XYZW or column not in segment.columns:
            continue
        out[column] = np.interp(tgt_s, src_s, _numeric_source(segment[column]))
    rotation = _resample_rotation(segment, src_s, tgt_s)
    for index, column in enumerate(ROTATION_XYZW):
        out[column] = rotation[:, index]

    # Label codes stay discrete: each new frame keeps the code in effect at that time.
    previous = np.searchsorted(src_s, tgt_s, side='right') - 1
    previous = np.clip(previous, 0, len(segment) - 1)
    out['auto_code'] = segment['auto_code'].to_numpy()[previous]
    out['subject'] = segment['subject'].iloc[0]
    out['study_num'] = segment['study_num'].iloc[0]

    tgt_time = t0 + pd.to_timedelta(tgt_s, unit='s')
    out['timestamp'] = _format_timestamps(tgt_time).to_numpy()
    out['milliseconds'] = np.rint(tgt_s * 1000.0).astype(np.int64)
    return out


def resample_session_to_30hz(df, filename=None, target_hz=TARGET_HZ, max_gap_s=MAX_GAP_S):
    """Resample one session so every later frame is 1/30 second.

    Sessions recorded near 60 Hz are brought down to 30 Hz. Sessions already
    near 30 Hz are placed on an exact 30 Hz grid, so 30, 90, 240, and 300
    frames cover 1, 3, 8, and 10 seconds in every session. Position is
    linearly interpolated and rotation is slerped. Label codes are held until
    the next code. A gap longer than `max_gap_s` starts a new segment instead
    of inventing the missing samples.
    """
    name = os.path.basename(filename) if filename else 'session'
    if 'timestamp' not in df.columns and 'Timestamp' in df.columns:
        df = df.rename(columns={'Timestamp': 'timestamp'})
    if 'timestamp' not in df.columns:
        raise ValueError('No timestamp column in {}; cannot resample to {} Hz'.format(name, target_hz))
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError('{} is missing pose columns: {}'.format(name, missing))

    n_raw = len(df)
    df = df.copy()
    df['_time'] = _parse_timestamps(df['timestamp'])
    n_bad = int(df['_time'].isna().sum())
    if n_bad:
        print('{}: dropping {} rows with unparseable timestamps'.format(name, n_bad))
        df = df.loc[df['_time'].notna()]
    df = df.sort_values('_time').drop_duplicates('_time', keep='first')
    if len(df) < 2:
        raise ValueError('{} does not have enough timestamps to resample'.format(name))

    gaps = df['_time'].diff().dt.total_seconds().fillna(0)
    segment_ids = (gaps > max_gap_s).cumsum().astype(int)
    parts = []
    for segment_id, segment in df.groupby(segment_ids, sort=True):
        resampled = _resample_segment(segment, target_hz)
        if resampled is None:
            continue
        resampled['num_sessions'] = int(segment_id)
        parts.append(resampled)
    if not parts:
        raise ValueError('{} has no segments long enough to resample to {} Hz'.format(name, target_hz))

    out = pd.concat(parts, ignore_index=True)
    n_segments = int(segment_ids.nunique())
    print('{}: {} raw rows -> {} rows at {} Hz across {} segment(s)'.format(
        name, n_raw, len(out), target_hz, n_segments
    ))
    return out


def load_gaze_change_set_data():
    """Pose, gaze, and max-distance channels, leaving out subjects with no gaze."""
    return load_change_set_data(include_gaze=True, exclude_subjects=GAZE_EXCLUDED_SUBJECTS)


def load_change_set_data(include_gaze=False, exclude_subjects=None):
    path = HEADSET_FINAL_DIR
    all_files = sorted(glob.glob(os.path.join(path, '**', '*.csv'), recursive=True))
    if not all_files:
        raise FileNotFoundError('No CSV files found under {}'.format(path))

    excluded = set(exclude_subjects or ())
    li = []
    for filename in all_files:
        subject = _extract_subject_id(filename)
        if subject in excluded:
            print('Skipping {} (subject {}); excluded from this dataset'.format(
                os.path.basename(filename), subject
            ), flush=True)
            continue
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
        if include_gaze:
            missing = [column for column in GAZE_COLUMNS if column not in df.columns]
            if missing:
                raise ValueError('{} is missing gaze columns: {}'.format(os.path.basename(filename), missing))
        df = _prepare_session_dataframe(df, filename)
        df = resample_session_to_30hz(df, filename)
        li.append(df)
    if not li:
        raise FileNotFoundError('No CSV files left under {} after exclusions {}'.format(
            path, sorted(excluded)
        ))

    sensor_data = pd.concat(li, ignore_index=True)
    sensor_data['unique_id'] = sensor_data['subject'].astype(str) + '_' + \
                               sensor_data['study_num'].astype(str) + '_' + sensor_data['num_sessions'].astype(str)
    print('Loaded {} files with subjects: {}'.format(
        len(li),
        sorted(sensor_data['subject'].unique().tolist()),
    ))

    label_mapping = {'p': 0, 'n': 1, 'd': 2}
    sensor_data['auto_code'] = sensor_data['auto_code'].replace(label_mapping)
    sensor_data['auto_code'] = sensor_data['auto_code'].replace(to_replace=r'^[^0-9]*$', value=2, regex=True)
    sensor_data['auto_code'] = sensor_data['auto_code'].apply(pd.to_numeric, errors='coerce')

    def _to_change_indicator(values):
        indicator = np.concatenate(([0], np.diff(values.to_numpy()) != 0)).astype(int)
        return pd.Series(indicator, index=values.index)

    sensor_data['auto_code'] = sensor_data.groupby('unique_id', sort=False)['auto_code'].transform(
        _to_change_indicator
    )
    # Change indicators are only 0/1; store as int64 so training targets match
    # nn.CrossEntropyLoss without a runtime cast.
    sensor_data['auto_code'] = sensor_data['auto_code'].astype(np.int64)
    print(sensor_data['auto_code'].value_counts())

    numeric_columns = list(FEATURES)
    if include_gaze:
        numeric_columns.extend(GAZE_COLUMNS)
    sensor_data[numeric_columns] = sensor_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    sensor_data.fillna(0, inplace=True)

    distance_signals = [
        (['headset_position_x', 'headset_position_y', 'headset_position_z'], 'headset_position'),
        (['headset_rotation_x', 'headset_rotation_y', 'headset_rotation_z'], 'headset_rotation'),
    ]
    if include_gaze:
        distance_signals.append((GAZE_ORIGIN_COLUMNS, 'gaze_origin'))
        distance_signals.append((GAZE_DIRECTION_COLUMNS, 'gaze_direction'))

    # 30, 90, and 300 frames are 1s, 3s, and 10s on the resampled 30 Hz grid.
    intervals = [TARGET_HZ, TARGET_HZ * 3, TARGET_HZ * 10]
    # Windows stay inside one session. A resampled dropout is a new unique_id,
    # and the following session must not inherit the previous pose.
    segment_ends = _segment_ends(sensor_data['unique_id'])
    args_list = [
        (sensor_data[columns].to_numpy(dtype=np.float64), interval, name, segment_ends)
        for interval in intervals
        for columns, name in distance_signals
    ]
    name_interval_pairs = [(name, interval) for interval in intervals for _, name in distance_signals]
    print('Computing max distances', flush=True)
    max_distances_list = [calculate_max_distance(args) for args in args_list]

    for (name, interval), max_distances in zip(name_interval_pairs, max_distances_list):
        column = 'max_{}_dist_{}s'.format(name, interval / TARGET_HZ)
        sensor_data[column] = max_distances
        print(column)

    # Center pose on each segment's median sitting pose after max-distance so
    # motion features stay translation-invariant while stored xyz/quat keep
    # absolute departures from the desk (standing, walking, turning).
    print('Centering headset pose on per-session median', flush=True)
    sensor_data = _center_pose_on_session_median(sensor_data)

    label_columns = ['auto_code', 'unique_id']
    for column in ('timestamp', 'milliseconds'):
        if column in sensor_data.columns:
            label_columns.append(column)
    label = sensor_data[label_columns].copy()
    if not include_gaze:
        sensor_data = sensor_data.drop(DROP_LIST_1, axis=1, errors='ignore')
        sensor_data = sensor_data.drop(DROP_LIST_2, axis=1, errors='ignore')
    input_columns = GAZE_MODEL_INPUT_COLUMNS if include_gaze else MODEL_INPUT_COLUMNS
    sensor_data = sensor_data[input_columns + ['unique_id', 'subject']]
    X_train, X_test, y_train, y_test = train_test_split(
        sensor_data,
        label,
        0.9,
        test_subjects=HEADSET_FINAL_TEST_SUBJECTS,
    )
    os.makedirs(FORMATTED_DIR, exist_ok=True)
    saved = {
        'train_x': change_data_path('train', 'x', gaze=include_gaze),
        'test_x': change_data_path('test', 'x', gaze=include_gaze),
        'train_y': change_data_path('train', 'y', gaze=include_gaze),
        'test_y': change_data_path('test', 'y', gaze=include_gaze),
    }
    X_train.to_csv(saved['train_x'], index=False)
    X_test.to_csv(saved['test_x'], index=False)
    y_train.to_csv(saved['train_y'], index=False)
    y_test.to_csv(saved['test_y'], index=False)
    for saved_path in saved.values():
        print('Wrote {}'.format(saved_path))

    print(sensor_data.head())
    print(sensor_data.keys())


def _segment_ends(unique_ids):
    ids = np.asarray(unique_ids)
    if len(ids) == 0:
        return np.array([], dtype=int)
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    return np.append(changes, len(ids))


def _align_quaternion_signs(quats):
    """Flip consecutive quaternions so each stays in the previous hemisphere."""
    out = np.asarray(quats, dtype=np.float64).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out


def _mean_unit_quaternion(quats):
    """Normalized mean of sign-aligned wxyz quaternions; identity if empty."""
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


def _quaternion_multiply_wxyz(left, right):
    """Hamilton product for wxyz quaternions. `left` may be (4,) or (N, 4)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim == 1:
        left = np.broadcast_to(left, right.shape)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), axis=-1)


def _center_pose_on_session_median(df):
    """Express headset pose relative to each session's median sitting pose.

    For every `unique_id` segment:
    - Position is shifted by the median xyz so the typical studying location is
      the origin. Departures (standing, walking) stay as large offsets.
    - Rotation is rewritten as `inverse(q_ref) * q_t`, where `q_ref` is the
      mean unit quaternion of the segment. Typical orientation becomes identity.

    Max-distance channels are left unchanged (already computed on absolute pose).
    """
    if 'unique_id' not in df.columns:
        raise ValueError('unique_id is required to center pose per session')

    pieces = []
    for _, group in df.groupby('unique_id', sort=False):
        group = group.copy()
        origin = group[POSITION_COLUMNS].median()
        group[POSITION_COLUMNS] = group[POSITION_COLUMNS] - origin

        quats = group[ROTATION_WXYZ].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(norms, 1e-8, None)
        q_ref = _mean_unit_quaternion(quats)
        q_ref_inv = q_ref * np.array([1.0, -1.0, -1.0, -1.0])
        relative = _quaternion_multiply_wxyz(q_ref_inv, quats)
        relative = np.where(relative[:, :1] < 0.0, -relative, relative)
        relative = _align_quaternion_signs(relative)
        group[ROTATION_WXYZ] = relative
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


@njit(parallel=True, cache=True)
def _window_max_euclidean_numba(values, interval):
    """Max pairwise Euclidean distance for every window of `interval` rows."""
    n_windows = values.shape[0] - interval + 1
    dims = values.shape[1]
    out = np.empty(n_windows, dtype=np.float64)
    for w in prange(n_windows):
        best = 0.0
        for i in range(interval - 1):
            base = w + i
            for j in range(i + 1, interval):
                other = w + j
                dist = 0.0
                for d in range(dims):
                    delta = values[base, d] - values[other, d]
                    dist += delta * delta
                if dist > best:
                    best = dist
        out[w] = np.sqrt(best)
    return out


def _window_max_euclidean(values, interval):
    """Max pairwise Euclidean distance in every window of `interval` rows.

    Same quantity as `pdist(window).max()`. A window of identical points returns 0.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.shape[0] < interval:
        return np.empty(0, dtype=np.float64)
    return _window_max_euclidean_numba(values, int(interval))


def calculate_max_distance(args):
    values, interval, name, segment_ends = args
    max_distances = np.zeros(len(values), dtype=np.float64)
    print('{} interval {} frames: {} rows'.format(name, interval, len(values)), flush=True)
    start = 0
    for end in segment_ends:
        length = int(end - start)
        if length >= interval:
            local = _window_max_euclidean(values[start:end], int(interval))
            # The first interval-1 frames have no full window yet, same as before.
            max_distances[start + interval - 1:end] = local
        start = end
    print('{} interval {} frames: done'.format(name, interval), flush=True)
    return max_distances
