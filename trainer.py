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

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = SummaryWriter(train_log_dir)

        # take over whatever gpus are on the system
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(config.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataloader if is_train else self.test_dataloader

            losses = []
            for it, (sample, label) in enumerate(loader):

                # place data on the correct device
                x = sample.to(config.device)
                y = label.to(config.device)
                hidden_state = torch.zeros([x.shape[0], config.hidden_state_size])

                for training_set in range(0, 256, config.timestep_interval):
                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        predictions, hidden_state = self.compute_forward(model, x, hidden_state, config, training_set)
                        predictions = torch.swapaxes(torch.stack(predictions), 0, 1)
                        local_label = y[:, training_set:training_set + 32]
                        cross_entropy_loss = self.cross_entropy_loss(predictions, local_label)
                        l2_loss = self.l2_regularization(model, config)
                        loss = cross_entropy_loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        loss += l2_loss
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        epoch_num += 1

                    hidden_state = hidden_state.detach()
                    self.summary_writer.add_scalar('losses/total', loss, epoch_num)
                    print(f"loss {loss} with epoch {epoch_num}")

            return epoch_num

        epoch = 0
        while True:
            epoch = run_epoch('train', epoch_num=epoch)

            if epoch > config.max_epochs:
                break

    def compute_forward(self, model, x, hidden_state, config, starting_idx):
        model_input = self.mask_timestep(x, starting_idx)
        predictions = []
        output = model(model_input, hidden_state)

        predictions.append(output["prediction"])

        for unroll_step in range(1, config.timestep_interval):
            hidden_state = output["hidden_state"]
            model_input = self.mask_timestep(x, starting_idx + unroll_step)
            output = model(model_input, hidden_state)

            predictions.append(output["prediction"])

        return predictions, hidden_state

    def cross_entropy_loss(self, prediction, target):
        return -(target * torch.nn.functional.log_softmax(prediction, -1)).sum(-1)

    def l2_regularization(self, model, config):
        return config.weight_decay * torch.sum(
            torch.stack([torch.sum(p ** 2.0) / 2
                         for p in model.parameters()])
        )

    def mask_timestep(self, x, t):
        masked_input = x.detach().clone()
        masked_input[:, t + 1:, :] = 0
        return masked_input
