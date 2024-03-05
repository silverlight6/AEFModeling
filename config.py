from dataclasses import dataclass

@dataclass
class modelConfig:
    sensor_count: int = 25
    max_timestep: int = 512
    time_window: int = max_timestep * 2
    sensor_patch_height: int = 64
    sensor_patch_width: int = 5
    image_patch_height: int = 16
    image_patch_width: int = 16
    sensor_channels: int = 1
    hidden_state_size: int = 128  # Default is 256
    num_heads: int = 2  # Default is 4
    num_hidden_layers: int = 2  # Default is 3
    mlp_layers: int = 2
    num_rnn_cells: int = 2
    embd_pdrop: float = 0.1
    lstm_size: int = int(hidden_state_size / (num_rnn_cells * 2))
    rnn_sizes = [lstm_size] * num_rnn_cells
    timestep_interval: int = 64
    max_epochs: int = 500
    batch_size: int = 256
    learning_rate = 0.003
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.0001
    # checkpoint settings
    ckpt_path = None
    class_weights = (0.0497149, 0.4530518, 0.4972332)

    device: str = 'cuda'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
