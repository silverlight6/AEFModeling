import numpy as np
import pandas as pd

def train_test_split(dataset, labels, split_amount):
    # Copying the list of participant_ids so I can separate by patient
    part_ids = dataset["subject"].copy()
    part_ids = list(set(part_ids))
    # Shuffling them
    np.random.shuffle(part_ids)

    # Create my training and test sets
    training, test = part_ids[:int(len(part_ids) * split_amount)], part_ids[int(len(part_ids) * split_amount):]
    train_x_prior = dataset[dataset["subject"].isin(training)]
    train_x_prior = train_x_prior.drop('subject', axis=1)
    test_x_prior = dataset[dataset["subject"].isin(test)]
    test_x_prior = test_x_prior.drop('subject', axis=1)
    train_x = train_x_prior.apply(pd.to_numeric, errors='coerce')
    test_x = test_x_prior.apply(pd.to_numeric, errors='coerce')
    train_x['unique_id'] = train_x_prior[['unique_id']].copy()
    test_x['unique_id'] = test_x_prior[['unique_id']].copy()
    train_x_index = train_x.index.values
    test_x_index = test_x.index.values

    # Create the labels for the training and test sets
    train_y = labels.iloc[train_x_index]
    test_y = labels.iloc[test_x_index]
    return train_x, test_x, train_y, test_y
