"""Range-of-motion clusters over fixed-length chunks of head motion.

Each session is cut into chunks of 1 s, 3 s and 10 s that overlap by half.
A chunk is expressed relative to its own first frame so the descriptors measure
motion inside that chunk (reach, turn, path length), not room location.
K-means groups the chunks of each length into motion types. Clusters are
numbered by activity: cluster 0 is the stillest and the last cluster is the
most active.

By default gaze is left out of the descriptors: several students have no gaze
recorded, so gaze features would mostly separate students instead of motions.
Pass ``use_gaze=True`` (and exclude those students) to cluster on gaze as well.
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features import CACHE_DIR, TARGET_HZ, _frame_signals, relative_rotvec, resampled_segments
from preprocessing.config import HEADSET_FINAL_DIR
from preprocessing.dataLoaders import _extract_subject_id

# Bump when the descriptors change so stale caches are rebuilt.
WINDOW_VERSION = 2
# Chunk lengths in frames: 1 s, 3 s and 10 s at 30 Hz. Chunks step by half a length.
SCALES = (30, 90, 300)
# Rotation axes follow Unity: x points right (pitch), y up (yaw), z forward (roll).
POSE_DESCRIPTORS = [
    'horizontal_reach_m',
    'vertical_range_m',
    'pitch_range_rad',
    'yaw_range_rad',
    'roll_range_rad',
    'turn_reach_rad',
    'path_length_m',
    'rotation_path_rad',
]
GAZE_DESCRIPTORS = [
    'gaze_path_rad',
    'gaze_peak_rad',
    'gaze_valid_share',
]
# Cache always stores pose + gaze descriptors; models choose which to use.
DESCRIPTORS = POSE_DESCRIPTORS + GAZE_DESCRIPTORS
# A cluster is switch-prone when its chunks contain a switch at least this
# many times as often as the average chunk of the same length.
SWITCH_PRONE_LIFT = 1.5


def scale_name(length):
    return '{:g}s'.format(length / TARGET_HZ)


def window_descriptors(pos, quat, speed, ang_speed, gaze_speed, gaze_valid, codes, length):
    """Start frames, range-of-motion descriptors and majority label code of each chunk."""
    n = len(pos)
    if n < length:
        return np.empty(0, dtype=int), np.empty((0, len(DESCRIPTORS))), np.empty(0, dtype=int)
    starts = np.arange(0, n - length + 1, length // 2)
    out = np.empty((len(starts), len(DESCRIPTORS)))
    majority = np.empty(len(starts), dtype=int)
    offsets = np.arange(length)
    n_codes = int(codes.max()) + 1
    chunk = max(1, 1_000_000 // length)
    for lo in range(0, len(starts), chunk):
        s = starts[lo:lo + chunk]
        idx = s[:, None] + offsets[None, :]
        rel = pos[idx] - pos[s][:, None, :]
        rotvec = relative_rotvec(quat[s][:, None, :], quat[idx])
        part = slice(lo, lo + len(s))
        out[part, 0] = np.linalg.norm(rel[..., [0, 2]], axis=-1).max(axis=1)
        out[part, 1] = rel[..., 1].max(axis=1) - rel[..., 1].min(axis=1)
        out[part, 2:5] = rotvec.max(axis=1) - rotvec.min(axis=1)
        out[part, 5] = np.linalg.norm(rotvec, axis=-1).max(axis=1)
        # speed[k] is the step from k - 1 to k, so the first frame adds nothing.
        out[part, 6] = speed[idx[:, 1:]].sum(axis=1) / TARGET_HZ
        out[part, 7] = ang_speed[idx[:, 1:]].sum(axis=1) / TARGET_HZ
        out[part, 8] = gaze_speed[idx[:, 1:]].sum(axis=1) / TARGET_HZ
        out[part, 9] = gaze_speed[idx].max(axis=1) / TARGET_HZ
        out[part, 10] = gaze_valid[idx].mean(axis=1)
        counts = (codes[idx][..., None] == np.arange(n_codes)).sum(axis=1)
        majority[part] = counts.argmax(axis=1)
    return starts, out, majority


def build_student_windows(path):
    tables = []
    for subject, segment_id, start_s, segment in resampled_segments(path):
        signals, pos, quat = _frame_signals(segment)
        codes = segment['auto_code'].to_numpy().astype(int)
        for length in SCALES:
            starts, desc, majority = window_descriptors(
                pos, quat,
                signals['speed'].to_numpy(),
                signals['ang_speed'].to_numpy(),
                signals['gaze_speed'].to_numpy(),
                signals['gaze_valid'].to_numpy(),
                codes, length,
            )
            if len(starts) == 0:
                continue
            table = pd.DataFrame(desc.astype(np.float32), columns=DESCRIPTORS)
            table.insert(0, 'subject', subject)
            table.insert(1, 'segment', segment_id)
            table.insert(2, 'scale', length)
            table.insert(3, 'window', np.arange(len(starts)))
            table.insert(4, 'start', starts)
            table.insert(5, 'time_s', start_s + starts / TARGET_HZ)
            table.insert(6, 'code', majority)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _cache_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, '{}_windows_v{}.pkl'.format(name, WINDOW_VERSION))


def _build_and_cache(path):
    build_student_windows(path).to_pickle(_cache_path(path))
    return path


def load_windows(rebuild=False, jobs=8, data_dir=HEADSET_FINAL_DIR, exclude_subjects=None):
    """Range-of-motion descriptors of every chunk of every student."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
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
        print('Building motion chunks for {} file(s)'.format(len(missing)))
        with Pool(processes=max(1, min(jobs, len(missing)))) as pool:
            for path in pool.imap_unordered(_build_and_cache, missing):
                print('  cached {}'.format(os.path.basename(path)))
    return pd.concat([pd.read_pickle(_cache_path(path)) for path in files], ignore_index=True)


def _contains_switch(windows, switches, lo_offset, hi_offset):
    """1 when a switch falls in [time_s + lo_offset, time_s + hi_offset) of each row."""
    out = np.zeros(len(windows), dtype=np.int8)
    for subject, rows in windows.groupby('subject', sort=False).indices.items():
        switch_t = switches.get(subject, np.empty(0))
        t = windows['time_s'].to_numpy()[rows]
        lo = np.searchsorted(switch_t, t + lo_offset, side='left')
        hi = np.searchsorted(switch_t, t + hi_offset, side='left')
        out[rows] = hi > lo
    return out


class MotionClusters:
    """K-means over range-of-motion descriptors, one model per chunk length."""

    def __init__(self, n_clusters=16, scales=SCALES, seed=0, use_gaze=False):
        self.n_clusters = n_clusters
        self.scales = tuple(scales)
        self.seed = seed
        self.use_gaze = bool(use_gaze)
        self.descriptors = list(DESCRIPTORS if self.use_gaze else POSE_DESCRIPTORS)
        self.medians, self.scalers, self.models, self.order = {}, {}, {}, {}

    def _transform(self, windows, length):
        # Motion ranges are heavy tailed. Dividing by the median and taking
        # log1p keeps small motions apart without letting big turns dominate.
        values = windows[self.descriptors].to_numpy(dtype=np.float64)
        return np.log1p(values / self.medians[length])

    def fit(self, windows):
        for length in self.scales:
            chunk = windows[windows['scale'] == length]
            self.medians[length] = np.maximum(
                np.median(chunk[self.descriptors].to_numpy(), axis=0), 1e-6
            )
            x = self._transform(chunk, length)
            self.scalers[length] = StandardScaler().fit(x)
            model = KMeans(self.n_clusters, n_init=4, random_state=self.seed)
            model.fit(self.scalers[length].transform(x))
            self.models[length] = model
            # Rank clusters by overall activity so cluster 0 is the stillest.
            rank = np.argsort(np.argsort(model.cluster_centers_.sum(axis=1)))
            self.order[length] = rank
        return self

    def assign(self, windows):
        """Cluster number of every chunk."""
        clusters = np.full(len(windows), -1, dtype=np.int32)
        for length in self.scales:
            rows = np.flatnonzero(windows['scale'].to_numpy() == length)
            if len(rows) == 0:
                continue
            x = self.scalers[length].transform(self._transform(windows.iloc[rows], length))
            clusters[rows] = self.order[length][self.models[length].predict(x)]
        return clusters

    def frame_features(self, rows, windows):
        """One-hot cluster features for each frame in `rows`.

        For every chunk length a frame gets the cluster of the last chunk that
        ends at or before it, the chunk centered on it, and the first chunk
        that starts at or after it, plus a flag for the before and after
        chunks being in different clusters.
        """
        frame = rows['frame'].to_numpy().astype(np.int64)
        features = {}
        for length in self.scales:
            chunk = windows[windows['scale'] == length].sort_values(['subject', 'segment', 'window'])
            counts = chunk.groupby(['subject', 'segment']).size()
            index = pd.DataFrame({'m': counts, 'offset': counts.cumsum() - counts}).reset_index()
            key = rows[['subject', 'segment']].merge(index, how='left', on=['subject', 'segment'])
            m = key['m'].fillna(0).to_numpy().astype(np.int64)
            offset = key['offset'].fillna(0).to_numpy().astype(np.int64)
            labels = chunk['cluster'].to_numpy()
            hop = length // 2

            def pick(position):
                valid = (position >= 0) & (position < m)
                out = np.full(len(frame), -1, dtype=np.int64)
                out[valid] = labels[offset[valid] + position[valid]]
                return out

            picked = {
                'before': pick(np.floor_divide(frame - length, hop)),
                'center': pick(np.clip(np.rint((frame - length / 2) / hop).astype(np.int64), 0, m - 1)),
                'after': pick(-np.floor_divide(-frame, hop)),
            }
            name = scale_name(length)
            for position, ids in picked.items():
                for k in range(self.n_clusters):
                    features['cluster_{}_{}_{}'.format(name, position, k)] = ids == k
            before, after = picked['before'], picked['after']
            features['cluster_{}_changed'.format(name)] = (before >= 0) & (after >= 0) & (before != after)
        return pd.DataFrame(features).astype(np.float32)

    def profiles(self, windows, switches):
        """What each cluster looks like and how often its chunks contain a switch."""
        tables = []
        for length in self.scales:
            chunk = windows[windows['scale'] == length].copy()
            chunk['has_switch'] = _contains_switch(chunk, switches, 0.0, length / TARGET_HZ)
            base_rate = chunk['has_switch'].mean()
            grouped = chunk.groupby('cluster')
            table = grouped[self.descriptors].median()
            table.insert(0, 'chunks', grouped.size())
            table.insert(1, 'share', table['chunks'] / len(chunk))
            table['switch_rate'] = grouped['has_switch'].mean()
            table['lift'] = table['switch_rate'] / base_rate if base_rate else np.nan
            table['switch_prone'] = table['lift'] >= SWITCH_PRONE_LIFT
            table['top_code'] = grouped['code'].agg(lambda c: int(c.mode().iloc[0]))
            table['top_code_share'] = grouped['code'].agg(lambda c: float((c == c.mode().iloc[0]).mean()))
            table.insert(0, 'scale', scale_name(length))
            tables.append(table.reset_index())
        return pd.concat(tables, ignore_index=True)

    def transitions(self, windows, switches, tolerance_s, min_count=50):
        """Switch rate at the boundary between one cluster and the next.

        Pairs are back-to-back chunks that do not overlap (chunk i and chunk
        i + 2, because chunks step by half a length). A boundary is near a
        switch when a switch is within `tolerance_s` of it.
        """
        tables = []
        for length in self.scales:
            chunk = windows[windows['scale'] == length].sort_values(['subject', 'segment', 'window'])
            nxt = chunk.groupby(['subject', 'segment'])['cluster'].shift(-2)
            pairs = chunk.assign(next_cluster=nxt, time_s=chunk['time_s'] + length / TARGET_HZ)
            pairs = pairs[pairs['next_cluster'].notna()]
            pairs['near_switch'] = _contains_switch(pairs, switches, -tolerance_s, tolerance_s + 1e-9)
            base_rate = pairs['near_switch'].mean()
            table = pairs.groupby(['cluster', 'next_cluster'])['near_switch'].agg(['size', 'mean'])
            table.columns = ['boundaries', 'switch_rate']
            table['lift'] = table['switch_rate'] / base_rate if base_rate else np.nan
            table = table[table['boundaries'] >= min_count].reset_index()
            table['next_cluster'] = table['next_cluster'].astype(int)
            table.insert(0, 'scale', scale_name(length))
            tables.append(table.sort_values('lift', ascending=False))
        return pd.concat(tables, ignore_index=True)
