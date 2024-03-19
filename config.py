from dataclasses import dataclass

@dataclass
class modelConfig:
    sensor_count: int = 26
    max_timestep: int = 512
    time_window: int = max_timestep * 2
    sensor_patch_height: int = 1
    sensor_patch_width: int = 26
    image_patch_height: int = 16
    image_patch_width: int = 16
    sensor_channels: int = 1
    hidden_state_size: int = 64  # Default is 256
    num_heads: int = 4  # Default is 4
    num_hidden_layers: int = 3  # Default is 3
    mlp_layers: int = 1
    num_rnn_cells: int = 2
    embd_pdrop: float = 0.1
    lstm_size: int = int(hidden_state_size / (num_rnn_cells * 2))
    rnn_sizes = [lstm_size] * num_rnn_cells
    timestep_interval: int = 32
    max_epochs: int = 500
    batch_size: int = 32
    learning_rate = 0.003
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.0001
    # checkpoint settings
    ckpt_path = None
    class_weights = (0.00011, 0.99989)
    # class_weights = (0.0497149, 0.4530518, 0.4972332)
    dataset_len_mult = 1

    # 9335.20930233 1.0001071328
    device: str = 'cuda'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


convTranConfig = {
    "save_dir": '/home/silver/MXTReality/AEF/Models/ConvCheckpoints',
    "pred_dir": '/home/silver/MXTReality/AEF/Models/Predictions',
    "tensorboard_dir": '/home/silver/MXTReality/AEF/Models/tensorboard_dir',
    "print_interval": 10,
    "emb_size": 16,  # Internal dimension of transformer embeddings
    "dim_ff": 256,  # Dimension of dense feedforward part of transformer layer
    "num_heads": 8,  # Number of multi-headed attention heads
    "Net_Type": 'C-T',
    "Fix_pos_encode": 'tAPE',
    "Rel_pos_encode": 'eRPE',
    "epochs": 100,
    "batch_size": 16,
    "lr": 1e-3,
    "dropout": 0.01,
    "val_interval": 2,
    "key_metric": 'precision',  # other options are loss and accuracy
    "gpu": 0,
    "console": 'store_true',
    "seed": 1234,
    "Data_shape": [16, 25, 512],
    "num_labels": 2,
    "device": 'cuda',
}
