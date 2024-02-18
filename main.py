from dataset import AEFDataset
from torch.utils.data import DataLoader
from aefModel import AefModel
from dataclasses import dataclass
from trainer import Trainer
import torch


@dataclass
class modelConfig:
    sensor_count: int = 13
    hidden_state_size: int = 256
    num_heads: int = 4
    num_hidden_layers: int = 3
    mlp_layers: int = 2
    num_rnn_cells: int = 2
    embd_pdrop: float = 0.1
    lstm_size: int = int(hidden_state_size / (num_rnn_cells * 2))
    rnn_sizes = [lstm_size] * num_rnn_cells
    max_timestep: int = 512
    timestep_interval: int = 32
    max_epochs: int = 20000
    batch_size: int = 64
    learning_rate = 3e-2
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.0001  # only applied on matmul weights
    # checkpoint settings
    ckpt_path = None

    device: str = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main():
    model_config = modelConfig()
    train_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_data_x.csv'
    train_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_data_y.csv'
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_data_y.csv'
    train_data = AEFDataset(train_x_path, train_y_path, model_config)
    test_data = AEFDataset(test_x_path, test_y_path, model_config)

    train_dataloader = DataLoader(train_data, batch_size=model_config.batch_size, num_workers=1)
    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, num_workers=1)

    model = AefModel(model_config)

    trainer = Trainer(model, train_dataloader, test_dataloader, model_config)

    trainer.train()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
