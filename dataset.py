import pandas as pd
import torch
import random
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
        self.max_timestep = config.max_timestep
        # print(f"number of batches in training set {len(self.model_input) / self.max_timestep}")

    def __len__(self):
        return len(self.unique_ids)

    # Now takes 0.04 seconds each call.
    def __getitem__(self, idx):
        # Fetch the student that we want
        student_id = self.unique_ids[idx]
        # Get all of that students sensor data
        selected_student_df = self.student_dataset.get_group(student_id)
        selected_student_df = selected_student_df.drop('unique_id', axis=1)

        # and it's associated label
        selected_label = self.student_labels.get_group(student_id)

        # Fetch the related data. If statement depending on where in the array we are accessing from
        random_index = random.randint(0, len(selected_student_df) - 2)
        if random_index > self.max_timestep and random_index + self.max_timestep < len(selected_student_df):
            sensor_data = selected_student_df.iloc[random_index:random_index + self.max_timestep]
            label_data = selected_label.iloc[random_index:random_index + self.max_timestep]
        elif random_index < self.max_timestep and random_index + self.max_timestep < len(selected_student_df):
            sensor_data = selected_student_df.iloc[:random_index]
            label_data = selected_label.iloc[:random_index]
        else:
            sensor_data = selected_student_df.iloc[random_index:]
            label_data = selected_label.iloc[random_index:]

        # This is the line that I am a little unsure about.
        # I don't want to do this for the entire dataset because we will not have the entire dataset during inference
        # but at the same time, there is often the same value repeated forever in some columns and this sort of
        # masks out any differences that could be seen there.
        # The flip argument is this helps with any device transfer because it ensures we will have the same range
        # of values regardless of the device that we are working with.
        # Downside is if everything is the same.. it creates null values..
        # sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.max() - sensor_data.min())

        # Pad with zeros if necessary (happens at the start and if we have less than 17 seconds of data)
        if len(sensor_data) < self.max_timestep:
            pad_length = self.max_timestep - len(sensor_data)
            padding_data = {col: [0] * pad_length for col in self.model_input.columns[:-1]}
            padding_label_data = {col: [1] * pad_length for col in self.labels.columns[:-1]}
            padding = pd.DataFrame(padding_data)
            padding_labels = pd.DataFrame(padding_label_data)
            padding = padding.drop('unique_id', axis=1)
            sensor_data = pd.concat([sensor_data, padding])
            label_data = pd.concat([label_data, padding_labels])

        label_data = label_data.drop('unique_id', axis=1)
        # Convert to tensors so I can use with the model.
        sensor_data = torch.tensor(sensor_data.values, dtype=torch.float32)
        label_data = torch.tensor(label_data.values, dtype=torch.float32)

        return sensor_data, label_data

