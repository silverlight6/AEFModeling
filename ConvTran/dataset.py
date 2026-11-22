import math
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset


# Headset pose in the formatted CSVs is already expressed relative to each
# session's median sitting pose (see preprocessing.dataLoaders). Position is a
# translation from that desk origin; rotation is relative to the session mean
# orientation (identity when seated as usual). Max-distance channels are
# unchanged absolute-motion features.
POSITION_COLUMNS = (
    'headset_position_x',
    'headset_position_y',
    'headset_position_z',
)
ROTATION_COLUMNS = (
    'headset_rotation_w',
    'headset_rotation_x',
    'headset_rotation_y',
    'headset_rotation_z',
)

# Training pose noise stays inside normal sitting-scale motion.
# Base bias ≈ seating / tracking offset; micro ≈ tremor and tiny posture drift.
POS_BIAS_STD_M = 0.015
POS_SINE_AMP_M = 0.004
POS_JITTER_STD_M = 0.0015
ROT_BIAS_STD_RAD = 0.03       # ~1.7 deg
ROT_SINE_AMP_RAD = 0.008      # ~0.5 deg
ROT_JITTER_STD_RAD = 0.003
SINE_FREQ_HZ = (0.15, 1.8)
# Same-label temporal splice (neg+neg or pos+pos). Off by default risk is low;
# enabled at a modest rate so most samples stay single-window.
SPLICE_PROB = 0.15


def _feature_columns(columns):
    return [column for column in columns if column != 'unique_id']


def position_channel_index(columns):
    """Channel indices of x, y, z after `unique_id` is removed."""
    feature_columns = _feature_columns(columns)
    return [feature_columns.index(name) for name in POSITION_COLUMNS]


def rotation_channel_index(columns):
    """Channel indices of w, x, y, z after `unique_id` is removed."""
    feature_columns = _feature_columns(columns)
    return [feature_columns.index(name) for name in ROTATION_COLUMNS]


def window_transition_label(label_values):
    """Binary label matching TercVJepa: positive only when exactly one transition is in the window.

    Zero transitions (stable) and two or more (transitioned back) are both negative.
    """
    count = int(np.sum(np.asarray(label_values).astype(bool)))
    return 1 if count == 1 else 0


def _quaternion_multiply(left, right):
    """Hamilton product of wxyz quaternions. Leading dims broadcast."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def _rotvec_to_quat(rotvec):
    """Convert axis-angle vectors (..., 3) to unit wxyz quaternions."""
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * angle
    axis = rotvec / angle
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def _unit_quaternions(quats):
    norm = quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return quats / norm


def apply_pose_noise(window, position_index, rotation_index, valid_length):
    """Add per-sample base bias plus low-frequency micro-motion on valid frames.

    `window` is (time, channels). Padding past `valid_length` is left at zero.
    Rotation noise is applied as small rotvecs via quaternion multiply so the
    result stays on the unit sphere.
    """
    if valid_length <= 0:
        return window

    window = window.clone()
    t = valid_length
    pos_idx = list(position_index)
    rot_idx = list(rotation_index)
    seconds = torch.arange(t, dtype=window.dtype, device=window.device) / 30.0

    # 1) Constant bias shared by every timestep in this sample.
    pos_bias = torch.randn(3, dtype=window.dtype, device=window.device) * POS_BIAS_STD_M
    window[:t, pos_idx] = window[:t, pos_idx] + pos_bias
    rot_bias = torch.randn(3, dtype=window.dtype, device=window.device) * ROT_BIAS_STD_RAD
    quats = _unit_quaternions(window[:t, rot_idx])
    quats = _quaternion_multiply(_rotvec_to_quat(rot_bias).expand_as(quats), quats)
    window[:t, rot_idx] = _unit_quaternions(quats)

    # 2) Micro-motion: independent slow sinusoid per axis + tiny Gaussian jitter.
    pos_micro = torch.zeros(t, 3, dtype=window.dtype, device=window.device)
    rot_micro = torch.zeros(t, 3, dtype=window.dtype, device=window.device)
    for axis in range(3):
        freq = random.uniform(*SINE_FREQ_HZ)
        phase = random.uniform(0.0, 2.0 * math.pi)
        wave = torch.sin(2.0 * math.pi * freq * seconds + phase)
        pos_micro[:, axis] = wave * random.uniform(0.0, POS_SINE_AMP_M)
        rot_micro[:, axis] = wave * random.uniform(0.0, ROT_SINE_AMP_RAD)
    pos_micro = pos_micro + torch.randn(t, 3, dtype=window.dtype, device=window.device) * POS_JITTER_STD_M
    rot_micro = rot_micro + torch.randn(t, 3, dtype=window.dtype, device=window.device) * ROT_JITTER_STD_RAD

    window[:t, pos_idx] = window[:t, pos_idx] + pos_micro
    quats = _unit_quaternions(window[:t, rot_idx])
    quats = _quaternion_multiply(_rotvec_to_quat(rot_micro), quats)
    window[:t, rot_idx] = _unit_quaternions(quats)
    return window


class AEFConvDataset(Dataset):
    def __init__(self, data_input_path, label_path, config, augment=True, sample_stride=None):
        # Reading the data in from the preprocessed files.
        self.model_input = pd.read_csv(data_input_path)
        # Filling in any NAN values that may have occurred in the process.
        # More of a precautionary step than anything else
        self.model_input.fillna(0, inplace=True)

        # I am fetching students by idx ID instead of idx of the full dataset to save compute
        self.unique_ids = self.model_input['unique_id'].unique()
        # Dataset to access to from the unique_ids
        self.student_dataset = self.model_input.groupby('unique_id')

        # Same as above but for labels.
        # Class indices must be int64 for nn.CrossEntropyLoss; values are only 0/1
        # after window_transition_label, so this cast is value-preserving.
        self.labels = pd.read_csv(label_path)
        self.labels['auto_code'] = pd.to_numeric(
            self.labels['auto_code'], errors='coerce'
        ).fillna(1).astype(np.int64)
        self.student_labels = self.labels.groupby('unique_id')
        # Separating this variable due to it's importance
        self.max_timestep = config.max_timestep
        # Training jitters the window start and adds pose noise every sample, so
        # each epoch sees a different version. Evaluation keeps the hashed start
        # and leaves the stored pose untouched.
        self.augment = augment
        self.sample_stride = self.max_timestep if sample_stride is None else sample_stride
        self.splice_prob = SPLICE_PROB

        self.hash_table = []
        self.build_hash_table()
        self.position_index = position_channel_index(self.model_input.columns)
        self.rotation_index = rotation_channel_index(self.model_input.columns)
        self.len_mult = config.dataset_len_mult

    def __len__(self):
        return len(self.hash_table) * self.len_mult

    def _window_label(self, unique_id, window_start):
        label_series = self.student_labels.get_group(unique_id)['auto_code']
        label_end = min(len(label_series), window_start + self.max_timestep)
        if window_start >= len(label_series):
            return 0
        return window_transition_label(label_series.iloc[window_start:label_end])

    def _load_window_frame(self, unique_id, window_start):
        """Return (time, channels) float32 array and how many real frames it has."""
        selected = self.student_dataset.get_group(unique_id).drop('unique_id', axis=1)
        if window_start + self.max_timestep < len(selected):
            frame = selected.iloc[window_start:window_start + self.max_timestep]
        else:
            frame = selected.iloc[window_start:]
        valid_length = len(frame)
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if valid_length < self.max_timestep:
            pad = np.zeros((self.max_timestep - valid_length, values.shape[1]), dtype=np.float32)
            values = np.concatenate([values, pad], axis=0) if valid_length else pad
        return values, valid_length

    def _maybe_splice(self, values, valid_length, label):
        """Join two same-label full windows at a random cut (third augmentation).

        Neg+neg stays a non-transition. Pos+pos is kept as positive even though
        the spliced clip may contain two change points — useful diversity for
        the rare positive class. Different labels and short/padded windows are
        skipped so padding never gets mixed into real frames.
        """
        if (
            valid_length < self.max_timestep
            or len(self.hash_table) < 2
            or random.random() >= self.splice_prob
        ):
            return values, valid_length

        other_hash = random.randrange(len(self.hash_table))
        other_id, other_idx = self.hash_table[other_hash]
        other_start = other_idx + random.randint(0, self.max_timestep - 1)
        if self._window_label(other_id, other_start) != label:
            return values, valid_length

        other_values, other_valid = self._load_window_frame(other_id, other_start)
        if other_valid < self.max_timestep:
            return values, valid_length

        cut = random.randint(self.max_timestep // 4, (3 * self.max_timestep) // 4)
        spliced = values.copy()
        spliced[:cut] = values[:cut]
        spliced[cut:] = other_values[cut:]
        return spliced, valid_length

    # TODO: create the label as a gradient. The closer I am to the change point, the closer my value is to 1.
    def __getitem__(self, idx):
        # Fetch the student that we want
        [student_id, student_idx] = self.hash_table[idx // self.len_mult]

        # DataLoader reseeds Python's random module in each worker at the start of every epoch,
        # so start jitter and pose noise differ on every pass through the training set.
        if self.augment:
            window_start = student_idx + random.randint(0, self.max_timestep - 1)
        else:
            window_start = student_idx

        label_data = self._window_label(student_id, window_start)
        values, valid_length = self._load_window_frame(student_id, window_start)

        if self.augment:
            values, valid_length = self._maybe_splice(values, valid_length, label_data)

        sensor_data = torch.tensor(values, dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.int64)

        if torch.isnan(sensor_data).any():
            sensor_data = torch.nan_to_num(sensor_data, 0)

        if self.augment and valid_length > 0:
            sensor_data = apply_pose_noise(
                sensor_data,
                self.position_index,
                self.rotation_index,
                valid_length,
            )
            # Re-zero padding so noise never leaks into unused frames.
            if valid_length < self.max_timestep:
                sensor_data[valid_length:] = 0

        sensor_data = torch.swapaxes(sensor_data, 0, 1)
        return sensor_data, label_tensor, idx // self.len_mult

    # Uncomment code to only include data that has NAN or off task labels
    def build_hash_table(self):
        for unique_id in self.unique_ids:
            student_data = self.student_dataset.get_group(unique_id)
            for idx in range(0, len(student_data), self.sample_stride):
                self.hash_table.append([unique_id, idx])
