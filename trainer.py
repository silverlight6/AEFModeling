import logging
import collections
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

Prediction = collections.namedtuple(
    'Prediction',
    'output hidden_state')


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config

        # These are used for logging to tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = SummaryWriter(train_log_dir)

        # take over whatever gpus are on the system
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(config.device)

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # Save the model so we can run tests and inference later.
        path = self.config.ckpt_path + f'aefmodel_{epoch}.pt'
        torch.save(raw_model.state_dict(), path)

    def train(self, epoch=0):
        model, config = self.model, self.config
        # model.module is used if using DataParallel (which I am on my set up due to multiple GPUs)
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            # Turns on dropout layers. That is all in this particular model.
            model.train(is_train)
            loader = self.train_dataloader if is_train else self.test_dataloader
            losses = []
            local_loss = 0
            loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights,
                                                                          dtype=torch.float32).to(config.device))
            for it, (sample, label) in enumerate(loader):
                # place data on the correct device
                x = sample.to(config.device)
                y = label.to(config.device)
                # hidden_state = torch.zeros([x.shape[0], config.hidden_state_size])

                for training_set in range(0, config.time_window, config.timestep_interval):
                    # collect gradients if training (no reason to if we aren't training)
                    with torch.set_grad_enabled(is_train):
                        # forward the model
                        # predictions, hidden_state = self.compute_forward_hidden(model, x, hidden_state,
                        #                                                         config, training_set)
                        predictions = self.compute_forward(model, x, config, training_set)
                        # Set up predictions so that they match the shape of the labels
                        predictions = torch.swapaxes(torch.swapaxes(torch.stack(predictions), 0, 1), 1, 2)
                        predictions_shape = predictions.shape
                        # Set up labels to match the shape of predictions
                        local_label = torch.reshape(y[:, training_set:training_set + config.timestep_interval],
                                                    shape=(predictions_shape[0], predictions_shape[2]))
                        # Calculate the loss. Using cross_entropy because it's standard. Not a requirement.
                        cross_entropy = loss_function(predictions, local_label)
                        # L2 regularization used to keep model weights from becoming too large
                        l2_loss = self.l2_regularization(model, config)
                        loss = cross_entropy.mean()  # collapse all losses if they are scattered on multiple gpus
                        loss += l2_loss
                        losses.append(loss.item())

                    local_loss += loss
                    if is_train:
                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        # gradient clipping used to keep model weights from becoming too large
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                    # hidden_state = hidden_state.detach()
                if is_train:
                    self.summary_writer.add_scalar('train_losses/total', local_loss, epoch_num)
                    print(f"loss {local_loss} with epoch {epoch_num}")
                else:
                    self.save_checkpoint(epoch)
                    self.summary_writer.add_scalar('test_losses/total', local_loss, epoch_num)
                    print(f"test_local_loss {local_loss} with epoch {epoch_num}")
                local_loss = 0
                epoch_num += 1

            # scheduler used because we are training for a long time
            # and lowering the learning rate over time tends to be a good thing
            scheduler.step()
            return epoch_num

        while True:
            epoch = run_epoch('train', epoch_num=epoch)
            print(f"finished one set through the dataset at epoch {epoch}")
            if epoch % 5 == 0:
                run_epoch('test', epoch_num=epoch)
            if epoch > config.max_epochs:
                break

    def compute_forward_hidden(self, model, x, hidden_state, config, starting_idx):
        model_input = self.mask_timestep(x, starting_idx)
        predictions = []

        output = model(model_input, hidden_state)

        predictions.append(output["prediction"])

        for unroll_step in range(1, config.timestep_interval):
            # Passing the hidden state from the previous time step to the next time step.
            hidden_state = output["hidden_state"]
            # Masking out all information that we do not have access to yet.
            model_input = self.mask_timestep(x, starting_idx + unroll_step)
            output = model(model_input, hidden_state)

            predictions.append(output["prediction"])

        return predictions, hidden_state

    def compute_forward(self, model, x, config, starting_idx):
        current_outputs = torch.zeros((config.batch_size, config.time_window, 1)).to(config.device)
        x = torch.cat((x, current_outputs), 2)
        model_input = self.mask_timestep(x, starting_idx)
        predictions = []

        output = model(model_input)

        predictions.append(output["prediction"])

        for unroll_step in range(1, config.timestep_interval):
            # Masking out all information that we do not have access to yet.
            x[:, starting_idx + unroll_step - 1, -1] = torch.argmax(output["prediction"], -1)
            model_input = self.mask_timestep(x, starting_idx + unroll_step)

            output = model(model_input)

            predictions.append(output["prediction"])

        return predictions

    def cross_entropy_loss(self, prediction, target):
        return -(target * torch.nn.functional.log_softmax(prediction, -1)).sum(-1)

    def l2_regularization(self, model, config):
        return config.weight_decay * torch.sum(
            torch.stack([torch.sum(p ** 2.0) / 2
                         for p in model.parameters()])
        )

    # This method masks out all time steps after the time step given.
    def mask_timestep(self, x, t):
        masked_input = x.detach().clone()
        masked_input[:, t + 1:, :] = 0
        if t < self.config.max_timestep:
            masked_input = masked_input[:, :self.config.max_timestep, :]
        else:
            masked_input = masked_input[:, t - self.config.max_timestep:t, :]
        return masked_input
