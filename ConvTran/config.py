import os
from dataclasses import dataclass

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

@dataclass
class modelConfig:

    batch_size: int = 32    # 8 seconds on the 30 Hz preprocessing grid (matches TercVJepa clip length).
    max_timestep: int = 240
    dataset_len_mult = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def apply_run_paths(config, gaze=False):
    """Copy `config` and point every artifact at the gaze run when `gaze` is set.

    The headset-only run keeps `ConvTran/outputs/`. The gaze run writes under
    `ConvTran/outputs/gaze/` and puts "gaze" in each checkpoint and prediction name.
    """
    config = dict(config)
    config['Data_shape'] = list(config['Data_shape'])
    if gaze:
        root = os.path.join(OUTPUT_DIR, 'gaze')
        config['checkpoint_prefix'] = 'AEF_gaze_model'
        config['best_checkpoint_name'] = 'gaze_model_best.pth'
        config['last_checkpoint_name'] = 'AEF_gaze_model_last.pth'
        config['prediction_name'] = 'headset_final_gaze_test_predictions.csv'
    else:
        root = OUTPUT_DIR
        config['checkpoint_prefix'] = 'AEF_model'
        config['best_checkpoint_name'] = 'model_best.pth'
        config['last_checkpoint_name'] = 'AEF_model_last.pth'
        config['prediction_name'] = 'headset_final_test_predictions.csv'
    config['save_dir'] = os.path.join(root, 'checkpoints')
    config['pred_dir'] = os.path.join(root, 'predictions')
    config['tensorboard_dir'] = os.path.join(root, 'tensorboard')
    return config


convTranConfig = {
    "save_dir": os.path.join(OUTPUT_DIR, 'checkpoints'),
    "pred_dir": os.path.join(OUTPUT_DIR, 'predictions'),
    "tensorboard_dir": os.path.join(OUTPUT_DIR, 'tensorboard'),
    "print_interval": 10,
    "emb_size": 16,  # Internal dimension of transformer embeddings
    "dim_ff": 256,  # Dimension of dense feedforward part of transformer layer
    "num_heads": 8,  # Number of multi-headed attention heads
    "Net_Type": 'C-T',
    "Fix_pos_encode": 'tAPE',
    "Rel_pos_encode": 'eRPE',
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "dropout": 0.01,
    "val_interval": 2,
    "key_metric": 'precision',  # positive-class precision; recall and f1 are logged too
    "checkpoint_interval": 50,
    "gpu": 0,
    "console": 'store_true',
    "seed": 1234,
    # "Data_shape": [32, 96, 10],
    "Data_shape": [32, 13, 240],
    "num_labels": 2,
    "device": 'cuda',
}
