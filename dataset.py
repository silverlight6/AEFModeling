import pandas as pd
import torch
from torch.utils.data import Dataset


class AEFDataset(Dataset):
    def __init__(self, data_input_path, label_path, config):
        self.model_input = pd.read_csv(data_input_path)
        self.model_input.fillna(0, inplace=True)

        self.labels = pd.read_csv(label_path)
        self.labels.fillna(0, inplace=True)
        self.max_timestep = config.max_timestep
        print(f"number of batches in training set {len(self.model_input) / self.max_timestep}")

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, idx):
        student_id = self.model_input.loc[idx, 'unique_id']
        same_pair_df = self.model_input[self.model_input['unique_id'] == student_id]
        same_pair_y = self.labels[self.labels['unique_id'] == student_id]

        sensor_data = same_pair_df.iloc[idx - same_pair_df.index[0]:idx - same_pair_df.index[0] + self.max_timestep]
        label_data = same_pair_y.iloc[idx - same_pair_y.index[0]:idx - same_pair_y.index[0] + self.max_timestep]

        # Pad with zeros if necessary
        if len(sensor_data) < self.max_timestep:
            pad_length = self.max_timestep - len(sensor_data)
            padding_data = {col: [0] * pad_length for col in self.model_input.columns[:-1]}
            padding_label_data = {col: [1] * pad_length for col in self.model_input.columns[:-1]}
            padding = pd.DataFrame(padding_data)
            padding_labels = pd.DataFrame(padding_label_data)
            sensor_data = pd.concat([sensor_data, padding])
            label_data = pd.concat([label_data, padding_labels])

        sensor_data = sensor_data.drop('unique_id', axis=1)
        label_data = label_data.drop('unique_id', axis=1)

        return torch.tensor(sensor_data.values, dtype=torch.float32), \
            torch.tensor(label_data.values, dtype=torch.float32)
