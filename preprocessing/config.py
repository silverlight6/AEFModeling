import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'Data'))
HEADSET_FINAL_DIR = os.path.join(DATA_ROOT, 'HeadsetFinal')
FORMATTED_DIR = os.path.join(DATA_ROOT, 'FormattedData')
HEADSET_FINAL_TEST_SUBJECTS = {31, 41}
# These sessions record gaze columns as zeros, so the gaze dataset leaves them out.
GAZE_EXCLUDED_SUBJECTS = {7, 8, 10, 17}


def change_data_path(split, axis, gaze=False):
    """Formatted change-point CSV. Gaze runs use a distinct filename."""
    kind = 'gaze_change_data' if gaze else 'change_data'
    return os.path.join(FORMATTED_DIR, '{}_{}_{}.csv'.format(split, kind, axis))
