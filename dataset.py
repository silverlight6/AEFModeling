import pandas as pd
import numpy as np
import torch
import random
import time
from torch.utils.data import Dataset


class AEFDataset(Dataset):
    def __init__(self, data_input_path, label_path, config):
        # Reading the data in from the preprocessed files.
        self.model_input = pd.read_csv(data_input_path)
        # Filling in any NAN values that may have occurred in the process.
        # More of a precautionary step than anything else
        self.model_input.fillna(0, inplace=True)

        # I am fetching students by idx ID instead of idx of the full dataset to save compute
        self.unique_ids = self.model_input['unique_id'].unique()
        # Dataset to access to from the unique_ids
        self.student_dataset = self.model_input.groupby('unique_id')

        # Same as above but for labels
        self.labels = pd.read_csv(label_path)
        self.labels.fillna(1, inplace=True)
        self.student_labels = self.labels.groupby('unique_id')
        # Separating this variable due to it's importance
        self.time_window = config.time_window

        self.hash_table = []
        self.build_hash_table()
        print(len(self.hash_table))
        # print(f"number of batches in training set {len(self.model_input) / self.max_timestep}")

        self.mean_noise = 0  # Mean of the Gaussian distribution
        self.std_dev_noise = 0.1  # Standard deviation of the Gaussian distribution

        self.len_mult = config.dataset_len_mult

    def __len__(self):
        return len(self.hash_table) * self.len_mult

    # Now takes 0.04 seconds each call.
    def __getitem__(self, idx):
        # Fetch the student that we want
        [student_id, student_idx] = self.hash_table[idx]
        # Get all of that students sensor data
        selected_student_df = self.student_dataset.get_group(student_id)
        selected_student_df = selected_student_df.drop('unique_id', axis=1)

        # and it's associated label
        selected_label = self.student_labels.get_group(student_id)

        # Fetch the related data. If statement depending on where in the array we are accessing from
        random_index = student_idx + random.randint(0, self.time_window - 1)
        if random_index + self.time_window < len(selected_student_df):
            sensor_data = selected_student_df.iloc[random_index:random_index + self.time_window]
            label_data = selected_label.iloc[random_index:random_index + self.time_window]

        else:
            sensor_data = selected_student_df.iloc[random_index:]
            label_data = selected_label.iloc[random_index:]

        # # Currently takes 0.03 seconds per call commenting out until I figure out a faster implementation later
        # for col in sensor_data.columns[:-1]:
        #     sensor_data.loc[:, col] = sensor_data.loc[:, col] + \
        #                        np.random.normal(self.mean_noise, self.std_dev_noise, size=len(sensor_data))

        # This is the line that I am a little unsure about.
        # I don't want to do this for the entire dataset because we will not have the entire dataset during inference
        # but at the same time, there is often the same value repeated forever in some columns and this sort of
        # masks out any differences that could be seen there.
        # The flip argument is this helps with any device transfer because it ensures we will have the same range
        # of values regardless of the device that we are working with.
        # Downside is if everything is the same.. it creates null values..
        # sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.max() - sensor_data.min())

        # Pad with zeros if necessary (happens at the start and if we have less than 17 seconds of data)
        if len(sensor_data) < self.time_window:
            pad_length = self.time_window - len(sensor_data)
            padding_data = {col: [0] * pad_length for col in self.model_input.columns[:-1]}
            padding_label_data = {col: [1] * pad_length for col in self.labels.columns[:-1]}
            padding = pd.DataFrame(padding_data)
            padding_labels = pd.DataFrame(padding_label_data)
            padding = padding.drop('unique_id', axis=1)
            sensor_data = pd.concat([sensor_data, padding])
            label_data = pd.concat([label_data, padding_labels])

        # label_data = label_data.drop('unique_id', axis=1)
        # Convert to tensors so I can use with the model.
        sensor_data = torch.tensor(sensor_data.values, dtype=torch.float32)
        label_data = torch.tensor(label_data.values, dtype=torch.long)

        # Keeping this here because the torch.tensor(sensor_data.values) line sometimes generates a NAN
        # On a very small subset of the data.
        if torch.isnan(sensor_data).any():
            sensor_data = torch.nan_to_num(sensor_data, 0)
        return sensor_data, label_data

    # Uncomment code to only include data that has NAN or off task labels
    def build_hash_table(self):
        # i = 0
        # j = 1
        for unique_id in self.unique_ids:
            student_data = self.student_dataset.get_group(unique_id)
            for idx in range(0, len(student_data), self.time_window):
                # if not (self.student_labels.get_group(unique_id)["auto_code"].iloc[idx:idx + self.max_timestep]
                #         == 0).all():
                #     self.hash_table.append([unique_id, idx])
                #     j += 1
                # else:
                #     i += 1
                self.hash_table.append([unique_id, idx])
        # print(f"I have {j} sections with a mix of labels and {i} sections with just on task behavior")


class AEFConvDataset(Dataset):
    def __init__(self, data_input_path, label_path, config):
        # Reading the data in from the preprocessed files.
        self.model_input = pd.read_csv(data_input_path)
        # Filling in any NAN values that may have occurred in the process.
        # More of a precautionary step than anything else
        self.model_input.fillna(0, inplace=True)

        # I am fetching students by idx ID instead of idx of the full dataset to save compute
        self.unique_ids = self.model_input['unique_id'].unique()
        # Dataset to access to from the unique_ids
        self.student_dataset = self.model_input.groupby('unique_id')

        # Same as above but for labels
        self.labels = pd.read_csv(label_path)
        self.labels.fillna(1, inplace=True)
        self.student_labels = self.labels.groupby('unique_id')
        # Separating this variable due to it's importance
        self.max_timestep = config.max_timestep

        self.hash_table = []
        self.build_hash_table()

        self.mean_noise = 0  # Mean of the Gaussian distribution
        self.std_dev_noise = 0.1  # Standard deviation of the Gaussian distribution
        self.two_d_label_window = 60
        self.len_mult = config.dataset_len_mult

    def __len__(self):
        return len(self.hash_table) * self.len_mult

    # TODO: create the label as a gradient. The closer I am to the change point, the closer my value is to 1.
    def __getitem__(self, idx):
        # Fetch the student that we want
        [student_id, student_idx] = self.hash_table[idx // self.len_mult]
        # Get all of that students sensor data
        selected_student_df = self.student_dataset.get_group(student_id)
        selected_student_df = selected_student_df.drop('unique_id', axis=1)

        # and it's associated label
        selected_label = self.student_labels.get_group(student_id)

        # Fetch the related data. If statement depending on where in the array we are accessing from
        random_index = student_idx + random.randint(0, self.max_timestep - 1)
        if random_index + self.max_timestep < len(selected_student_df):
            sensor_data = selected_student_df.iloc[random_index:random_index + self.max_timestep]
            label_data = selected_label.iloc[random_index + self.max_timestep - self.two_d_label_window:
                                             random_index + self.max_timestep].any().astype(int).values[0]
        else:
            sensor_data = selected_student_df.iloc[random_index:]
            label_data = selected_label.iloc[-self.two_d_label_window:].any().astype(int).values[0]

        # Pad with zeros if necessary (happens at the start and if we have less than 17 seconds of data)
        if len(sensor_data) < self.max_timestep:
            pad_length = self.max_timestep - len(sensor_data)
            padding_data = {col: [0] * pad_length for col in self.model_input.columns[:-1]}
            padding = pd.DataFrame(padding_data)
            padding = padding.drop('unique_id', axis=1)
            sensor_data = pd.concat([sensor_data, padding])

        # Convert to tensors so I can use with the model.
        sensor_data = torch.tensor(sensor_data.values, dtype=torch.float32)
        # new_label = [0.0, 0.0]
        # new_label[label_data] = 1.0
        # label_data = torch.tensor(new_label, dtype=torch.float32)
        label_data = torch.tensor(label_data, dtype=torch.int8)
        # print(label_data)

        # Keeping this here because the torch.tensor(sensor_data.values) line sometimes generates a NAN
        # On a very small subset of the data.
        if torch.isnan(sensor_data).any():
            sensor_data = torch.nan_to_num(sensor_data, 0)
        sensor_data = torch.swapaxes(sensor_data, 0, 1)
        return sensor_data, label_data, idx // self.len_mult

    # Uncomment code to only include data that has NAN or off task labels
    def build_hash_table(self):
        # i = 0
        # j = 1
        for unique_id in self.unique_ids:
            student_data = self.student_dataset.get_group(unique_id)
            for idx in range(0, len(student_data), self.max_timestep):
                # if not (self.student_labels.get_group(unique_id)["auto_code"].iloc[idx:idx + self.max_timestep]
                #         == 0).all():
                #     self.hash_table.append([unique_id, idx])
                #     j += 1
                # else:
                #     i += 1
                self.hash_table.append([unique_id, idx])
        # print(f"I have {j} sections with a mix of labels and {i} sections with just on task behavior")

