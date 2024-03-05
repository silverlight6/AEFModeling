import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import AEFDataset
from aefModel import AefModel
from sklearn.metrics import confusion_matrix
from dataVisuals.visualizer import show_confustion_matrix
from config import modelConfig


def test_results():
    model_config = modelConfig()
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_data_y.csv'

    PATH = '/home/silver/MXTReality/AEF/Models/HiddenStateModels/aefmodel_104.pt'

    test_data = AEFDataset(test_x_path, test_y_path, model_config)

    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)

    model = AefModel(model_config)

    model.load_state_dict(torch.load(PATH))
    test(model, modelConfig, test_dataloader)


def test(model, config, loader):
    # model.module is used if using DataParallel (which I am on my set up due to multiple GPUs)
    model = torch.nn.DataParallel(model).to(config.device)
    pred_cm = None
    label_cm = None

    def run_epoch(epoch_num=0):
        # Turns on dropout layers. That is all in this particular model.
        losses = []
        local_loss = 0
        total_pred = None
        total_label = None
        loss_function = torch.nn.CrossEntropyLoss()
        for it, (sample, label) in enumerate(loader):
            # place data on the correct device
            x = sample.to(config.device)
            y = label.to(config.device)
            hidden_state = torch.zeros([x.shape[0], config.hidden_state_size])

            for training_set in range(0, config.max_timestep, config.timestep_interval):
                with torch.set_grad_enabled(False):
                    # forward the model
                    predictions, hidden_state = compute_forward_hidden(model, x, hidden_state, config, training_set)
                    # predictions = compute_forward(model, x, config, training_set)
                    # Set up predictions so that they match the shape of the labels
                    predictions = torch.swapaxes(torch.swapaxes(torch.stack(predictions), 0, 1), 1, 2)
                    predictions_shape = predictions.shape
                    # Set up labels to match the shape of predictions
                    local_label = torch.reshape(y[:, training_set:training_set + config.timestep_interval],
                                                shape=(predictions_shape[0], predictions_shape[2]))
                    # Calculate the loss. Using cross_entropy because it's standard. Not a requirement.
                    # print(f"predictions {predictions}")
                    cross_entropy = loss_function(predictions, local_label)
                    # L2 regularization used to keep model weights from becoming too large
                    l2_loss = l2_regularization(model, config)
                    loss = cross_entropy.mean()  # collapse all losses if they are scattered on multiple gpus
                    loss += l2_loss
                    losses.append(loss.item())
                    print(predictions_shape)

                    if total_pred is None:
                        total_pred = torch.softmax(predictions, dim=1).cpu().numpy()
                        total_label = local_label.cpu().numpy()
                    else:
                        total_pred = np.concatenate((total_pred, torch.softmax(predictions, dim=1).cpu().numpy()),
                                                    axis=0)
                        total_label = np.concatenate((total_label, local_label.cpu().numpy()), axis=0)
                local_loss += loss

                hidden_state = hidden_state.detach()
            print(f"test_local_loss {local_loss} with epoch {epoch_num}")
            local_loss = 0
            epoch_num += 1

        # TODO: Make accumulate over batches
        # [number of intervals, batch size, classes, timeframe]
        total_pred = np.argmax(total_pred, axis=1)
        total_pred = np.reshape(total_pred, (-1,))
        total_label = np.reshape(total_label, (-1,))

        return epoch_num, total_pred, total_label

    epoch = 0
    while True:
        epoch, pred, label = run_epoch(epoch_num=epoch)
        if pred_cm is None:
            pred_cm = pred
            label_cm = label
        else:
            pred_cm = np.concatenate((pred_cm, pred), axis=0)
            label_cm = np.concatenate((label_cm, label), axis=0)
        if epoch > 2:
            break
    print(pred_cm.shape)
    print(label_cm.shape)
    cm = confusion_matrix(pred_cm.tolist(), label_cm.tolist())
    show_confustion_matrix(cm)

# TODO: Switch dataset to allow for longer time steps
def compute_forward_hidden(model, x, hidden_state, config, starting_idx):
    model_input = mask_timestep(x, starting_idx, config.max_timestep)
    predictions = []

    output = model(model_input, hidden_state)

    predictions.append(output["prediction"])

    for unroll_step in range(1, config.timestep_interval):
        # Passing the hidden state from the previous time step to the next time step.
        hidden_state = output["hidden_state"]
        # Masking out all information that we do not have access to yet.
        model_input = mask_timestep(x, starting_idx + unroll_step, config.max_timestep)
        output = model(model_input, hidden_state)

        predictions.append(output["prediction"])

    return predictions, hidden_state

def compute_forward(model, x, config, starting_idx):
    model_input = mask_timestep(x, starting_idx, config.max_timestep)
    predictions = []

    output = model(model_input)

    predictions.append(output["prediction"])

    for unroll_step in range(1, config.timestep_interval):
        # Masking out all information that we do not have access to yet.
        model_input = mask_timestep(x, starting_idx + unroll_step, config.max_timestep)
        output = model(model_input)

        predictions.append(output["prediction"])

    return predictions

def cross_entropy_loss(prediction, target):
    return -(target * torch.nn.functional.log_softmax(prediction, -1)).sum(-1)

def l2_regularization(model, config):
    return config.weight_decay * torch.sum(
        torch.stack([torch.sum(p ** 2.0) / 2
                     for p in model.parameters()])
    )

# This method masks out all time steps after the time step given.
def mask_timestep(x, t, max_timestep):
    masked_input = x.detach().clone()
    masked_input[:, t + 1:, :] = 0
    if t < max_timestep:
        masked_input = masked_input[:, :max_timestep, :]
    else:
        masked_input = masked_input[:, t - max_timestep:t, :]
    return masked_input
