"""Random forest that flags frames within 30 frames (1 s) of a label switch.

Usage (from TimeSeriesModel/):
    venv/bin/python Forest/train_forest.py
    venv/bin/python Forest/train_forest.py --window-frames 30 --cv-folds 5 --n-estimators 300
    venv/bin/python Forest/train_forest.py --gaze --features both --clusters 16 --out Forest/output_gaze_both16

Every change of label code is a switch. Frames up to `--window-frames` before
or after a switch are positive. At evaluation the per-frame probabilities are
turned into switch detections (peaks above a threshold, at least one window
apart), and a detection is correct when it lands within `--window-frames` of a
real switch in either direction. Each switch can be matched once.

By default the forest only sees motion clusters (see clusters.py): sessions
are cut into 1 s, 3 s and 10 s chunks, each chunk is assigned a range-of-motion
cluster, and every frame is described by the clusters before, around and after
it. `--features motion` uses the hand-built window statistics from features.py
instead, and `--features both` uses both.

Training students are split into leave-students-out folds to pick the
threshold with the best switch-level F1. The final forest is trained on every
training student and scored on the held-out test students from preprocessing/config.py.
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold  # noqa: E402

from clusters import MotionClusters, load_windows  # noqa: E402
from features import TARGET_HZ, load_dataset, switch_times  # noqa: E402
from preprocessing.config import GAZE_EXCLUDED_SUBJECTS, HEADSET_FINAL_TEST_SUBJECTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Half a frame of slack absorbs millisecond rounding in the timestamps.
SLACK_S = 0.5 / TARGET_HZ


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--window-frames', type=int, default=30,
                        help='Frames on each side of a switch that are positive and that a '
                             'detection may be off by (30 frames = 1 s).')
    parser.add_argument('--features', choices=['clusters', 'motion', 'both'], default='clusters',
                        help='Forest inputs: motion-cluster memberships, window statistics, or both.')
    parser.add_argument('--clusters', type=int, default=16,
                        help='Range-of-motion clusters per chunk length.')
    parser.add_argument('--stride', type=int, default=3,
                        help='Keep every Nth frame of the training students.')
    parser.add_argument('--test-stride', type=int, default=1,
                        help='Keep every Nth frame of the test students.')
    parser.add_argument('--neg-ratio', type=float, default=20.0,
                        help='Negatives kept per positive when fitting. 0 keeps every negative.')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--min-samples-leaf', type=int, default=5)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--max-features', default='sqrt')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Leave-students-out folds used to choose the threshold. Below 2 uses 0.5.')
    parser.add_argument('--test-subjects', type=int, nargs='+', default=sorted(HEADSET_FINAL_TEST_SUBJECTS))
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument(
        '--gaze',
        action='store_true',
        help='Drop students without gaze ({}), put gaze into motion clusters, '
             'and keep gaze window features.'.format(sorted(GAZE_EXCLUDED_SUBJECTS)),
    )
    parser.add_argument('--out', default=None,
                        help='Output directory. Defaults to Forest/output or Forest/output_gaze.')
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, 'output_gaze' if args.gaze else 'output')
    return args


def make_forest(args):
    max_features = args.max_features
    try:
        max_features = float(max_features)
    except ValueError:
        pass
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        max_features=max_features,
        class_weight='balanced_subsample',
        n_jobs=args.jobs,
        random_state=args.seed,
    )


def stride_sample(data, stride):
    return data[data['frame'] % stride == 0] if stride > 1 else data


def subsample_negatives(y, ratio, rng):
    """Row indices with every positive and at most `ratio` negatives per positive."""
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if ratio > 0 and len(negatives) > ratio * len(positives):
        negatives = rng.choice(negatives, int(ratio * len(positives)), replace=False)
    return np.sort(np.concatenate([positives, negatives]))


def fit(args, X, y, rng):
    rows = subsample_negatives(y, args.neg_ratio, rng)
    forest = make_forest(args)
    forest.fit(X[rows], y[rows])
    return forest


def frame_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        'threshold': float(threshold),
        'frames': int(len(y)),
        'positive_rate': float(np.mean(y)),
        'roc_auc': float(roc_auc_score(y, prob)) if both_classes else None,
        'average_precision': float(average_precision_score(y, prob)) if both_classes else None,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(np.mean(pred == y)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


class SwitchEvaluator:
    """Scores per-frame probabilities as switch detections with a time tolerance."""

    def __init__(self, data, switches, window_frames, stride):
        self.tolerance_s = window_frames / TARGET_HZ
        self.radius = max(1, window_frames // stride)
        self.switches = switches
        self.hours = len(data) * stride / TARGET_HZ / 3600.0
        self.times = data['time_s'].to_numpy()
        self.groups = list(data.groupby(['subject', 'segment'], sort=False).indices.values())
        self.subject_of_row = data['subject'].to_numpy()

    def detect(self, prob, threshold):
        """Row positions of detected switches.

        A detection is a frame above `threshold` that is the maximum within
        one tolerance on each side. Two such frames inside one tolerance can
        only be a tie, so only the first of them is kept.
        """
        detections = []
        for rows in self.groups:
            p = prob[rows]
            peak = maximum_filter1d(p, size=2 * self.radius + 1, mode='nearest')
            last = -np.inf
            for c in np.flatnonzero((p >= threshold) & (p == peak)):
                if c - last > self.radius:
                    detections.append(rows[c])
                    last = c
        return np.asarray(detections, dtype=int)

    def _matches(self, switch_t, detect_t):
        """One-to-one pairs of switches and detections within the tolerance, closest first."""
        if len(switch_t) == 0 or len(detect_t) == 0:
            return 0, []
        detect_t = np.sort(detect_t)
        pairs = []
        for i, s in enumerate(switch_t):
            lo = np.searchsorted(detect_t, s - self.tolerance_s - SLACK_S, side='left')
            hi = np.searchsorted(detect_t, s + self.tolerance_s + SLACK_S, side='right')
            pairs.extend((abs(detect_t[j] - s), i, j) for j in range(lo, hi))
        pairs.sort()
        used_switch, used_detect, offsets = set(), set(), []
        for distance, i, j in pairs:
            if i not in used_switch and j not in used_detect:
                used_switch.add(i)
                used_detect.add(j)
                offsets.append(detect_t[j] - switch_t[i])
        return len(offsets), offsets

    def score(self, prob, threshold, detections=None):
        if detections is None:
            detections = self.detect(prob, threshold)
        detect_subjects = self.subject_of_row[detections]
        matched = total_switches = 0
        offsets = []
        for subject, switch_t in self.switches.items():
            hits, subject_offsets = self._matches(switch_t, self.times[detections[detect_subjects == subject]])
            matched += hits
            offsets.extend(subject_offsets)
            total_switches += len(switch_t)
        n_detect = len(detections)
        precision = matched / n_detect if n_detect else 0.0
        recall = matched / total_switches if total_switches else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            'threshold': float(threshold),
            'tolerance_frames': int(round(self.tolerance_s * TARGET_HZ)),
            'switches': int(total_switches),
            'detections': int(n_detect),
            'matched': int(matched),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'false_alarms_per_hour': float((n_detect - matched) / self.hours) if self.hours else None,
            'mean_abs_offset_s': float(np.mean(np.abs(offsets))) if offsets else None,
            'hours': float(self.hours),
        }

    def best_threshold(self, prob):
        """Threshold with the best switch-level F1, searched over the upper tail of `prob`."""
        grid = np.unique(np.round(np.quantile(prob, np.linspace(0.5, 0.9999, 150)), 4))
        scores = [self.score(prob, threshold) for threshold in grid]
        best = max(scores, key=lambda s: (s['f1'], s['threshold']))
        return best['threshold'], scores


def save_plots(path, curves, sweeps):
    fig, (frame_ax, event_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for name, (y, prob) in curves.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        frame_ax.plot(recall, precision, label='{} (AP={:.3f})'.format(name, average_precision_score(y, prob)))
        frame_ax.axhline(np.mean(y), linestyle=':', color=frame_ax.lines[-1].get_color(), linewidth=1)
    frame_ax.set_xlabel('Recall')
    frame_ax.set_ylabel('Precision')
    frame_ax.set_title('Frame level (dotted = base rate)')
    frame_ax.legend()

    for name, scores in sweeps.items():
        thresholds = [s['threshold'] for s in scores]
        line, = event_ax.plot(thresholds, [s['f1'] for s in scores], label='{} F1'.format(name))
        event_ax.plot(thresholds, [s['precision'] for s in scores], linestyle='--', color=line.get_color(),
                      label='{} precision'.format(name))
        event_ax.plot(thresholds, [s['recall'] for s in scores], linestyle=':', color=line.get_color(),
                      label='{} recall'.format(name))
    event_ax.set_xlabel('Threshold')
    event_ax.set_title('Switch level, +/- {} frames'.format(scores[0]['tolerance_frames'] if sweeps else ''))
    event_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    exclude_subjects = GAZE_EXCLUDED_SUBJECTS if args.gaze else None
    if args.gaze:
        print('Gaze run: excluding subjects {} and clustering with gaze descriptors'.format(
            sorted(GAZE_EXCLUDED_SUBJECTS)))
    data, motion_columns = load_dataset(
        args.window_frames,
        args.rebuild_cache,
        args.jobs,
        exclude_subjects=exclude_subjects,
        include_gaze_features=True,
    )
    switches = {
        subject: switch_times(group['time_s'], group['code'])
        for subject, group in data.groupby('subject', sort=False)
    }
    is_test = data['subject'].isin(args.test_subjects)
    train = stride_sample(data[~is_test], args.stride).reset_index(drop=True)
    test = stride_sample(data[is_test], args.test_stride).reset_index(drop=True)
    del data
    train_switches = {s: t for s, t in switches.items() if s not in args.test_subjects}
    test_switches = {s: t for s, t in switches.items() if s in args.test_subjects}

    feature_columns = motion_columns if args.features != 'clusters' else []
    X_train = train[feature_columns].to_numpy(dtype=np.float32) if feature_columns else np.empty(
        (len(train), 0), dtype=np.float32
    )
    X_test = test[feature_columns].to_numpy(dtype=np.float32) if feature_columns else np.empty(
        (len(test), 0), dtype=np.float32
    )
    motion = None
    if args.features != 'motion':
        windows = load_windows(
            args.rebuild_cache, args.jobs, exclude_subjects=exclude_subjects
        )
        train_windows = windows['subject'].isin(list(train_switches)).to_numpy()
        motion = MotionClusters(
            args.clusters, seed=args.seed, use_gaze=args.gaze
        ).fit(windows[train_windows])
        windows['cluster'] = motion.assign(windows)
        tolerance_s = args.window_frames / TARGET_HZ
        profiles = motion.profiles(windows[train_windows], train_switches)
        profiles.to_csv(os.path.join(args.out, 'cluster_profiles.csv'), index=False)
        motion.transitions(windows[train_windows], train_switches, tolerance_s).to_csv(
            os.path.join(args.out, 'cluster_transitions.csv'), index=False)
        print('Motion clusters (training students):')
        profile_cols = ['scale', 'cluster', 'share', 'yaw_range_rad', 'path_length_m',
                        'switch_rate', 'lift', 'top_code']
        if args.gaze:
            profile_cols[4:4] = ['gaze_path_rad', 'gaze_valid_share']
        print(profiles[profile_cols].to_string(index=False))
        cluster_train = motion.frame_features(train, windows)
        cluster_test = motion.frame_features(test, windows)
        feature_columns = feature_columns + list(cluster_train.columns)
        X_train = np.hstack([X_train, cluster_train.to_numpy()])
        X_test = np.hstack([X_test, cluster_test.to_numpy()])
        del windows, cluster_train, cluster_test

    y_train = train['label'].to_numpy()
    y_test = test['label'].to_numpy()
    print('Features: {} ({})'.format(len(feature_columns), args.features))
    print('Train: {} frames from {} students, {} switches, {:.2%} of frames near a switch'.format(
        len(y_train), train['subject'].nunique(), sum(map(len, train_switches.values())), y_train.mean()))
    print('Test:  {} frames from students {}, {} switches, {:.2%} of frames near a switch'.format(
        len(y_test), sorted(test['subject'].unique().tolist()),
        sum(map(len, test_switches.values())), y_test.mean()))

    metrics = {'args': vars(args), 'features': feature_columns}
    threshold = 0.5
    curves, sweeps = {}, {}
    groups = train['subject'].to_numpy()
    if args.cv_folds >= 2 and len(np.unique(groups)) >= args.cv_folds:
        oof = np.zeros(len(y_train))
        for fold, (fit_rows, held_rows) in enumerate(GroupKFold(args.cv_folds).split(X_train, y_train, groups)):
            forest = fit(args, X_train[fit_rows], y_train[fit_rows], rng)
            oof[held_rows] = forest.predict_proba(X_train[held_rows])[:, 1]
            held_subjects = sorted(np.unique(groups[held_rows]).tolist())
            print('CV fold {}: held out {} | frame AP {:.3f}'.format(
                fold + 1, held_subjects, average_precision_score(y_train[held_rows], oof[held_rows])))
        cv_events = SwitchEvaluator(train, train_switches, args.window_frames, args.stride)
        threshold, cv_sweep = cv_events.best_threshold(oof)
        metrics['cv'] = {
            'frames': frame_metrics(y_train, oof, threshold),
            'switches': cv_events.score(oof, threshold),
        }
        curves['CV'] = (y_train, oof)
        sweeps['CV'] = cv_sweep
        print('CV best switch-level F1 threshold: {:.4f}'.format(threshold))

    forest = fit(args, X_train, y_train, rng)
    prob = forest.predict_proba(X_test)[:, 1]
    test_events = SwitchEvaluator(test, test_switches, args.window_frames, args.test_stride)
    detections = test_events.detect(prob, threshold)
    metrics['test'] = {
        'frames': frame_metrics(y_test, prob, threshold),
        'switches': test_events.score(prob, threshold, detections),
    }
    curves['Test {}'.format(args.test_subjects)] = (y_test, prob)
    _, sweeps['Test'] = test_events.best_threshold(prob)
    metrics['seconds'] = time.time() - started

    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({
        'model': forest,
        'motion_clusters': motion,
        'feature_columns': feature_columns,
        'threshold': threshold,
        'window_frames': args.window_frames,
    }, os.path.join(args.out, 'forest.joblib'), compress=3)
    pd.DataFrame({
        'feature': feature_columns,
        'importance': forest.feature_importances_,
    }).sort_values('importance', ascending=False).to_csv(
        os.path.join(args.out, 'feature_importances.csv'), index=False)
    detected = np.zeros(len(test), dtype=np.int8)
    detected[detections] = 1
    test[['subject', 'segment', 'frame', 'time_s', 'code', 'label']].assign(
        probability=prob, detection=detected
    ).to_csv(os.path.join(args.out, 'test_predictions.csv'), index=False)
    save_plots(os.path.join(args.out, 'pr_curve.png'), curves, sweeps)

    summary = {k: v for k, v in metrics.items() if k in ('cv', 'test')}
    print(json.dumps(summary, indent=2))
    print('Saved model and reports to {}'.format(args.out))


if __name__ == '__main__':
    main()
