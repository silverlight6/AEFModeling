import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .Models.loss import l2_reg_loss
from .Models import utils, analysis

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less
POSITIVE_CLASS = 1  # transition class


def positive_class_scores(precision, recall, class_ind, positive_class=POSITIVE_CLASS):
    """Precision, recall, and F1 for the transition class.

    `precision` and `recall` follow `existing_class_ind`, which omits classes
    absent from both the labels and the predictions. A missing positive class
    scores 0, matching an all-negative epoch.
    """
    class_ind = list(class_ind)
    if positive_class in class_ind:
        idx = class_ind.index(positive_class)
        pos_precision = float(precision[idx])
        pos_recall = float(recall[idx])
    else:
        pos_precision = 0.0
        pos_recall = 0.0
    if not np.isfinite(pos_precision):
        pos_precision = 0.0
    if not np.isfinite(pos_recall):
        pos_recall = 0.0
    denom = pos_precision + pos_recall
    f1 = 0.0 if denom == 0 else (2.0 * pos_precision * pos_recall / denom)
    return pos_precision, pos_recall, f1


class BaseTrainer(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat =False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super(SupervisedTrainer, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            # self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=False, plot=False)
        else:
            self.classification = False
        if kwargs['print_conf_mat']:
            # Print the matrix only. plot=True calls plt.show(block=True) and would hang after training.
            self.analyzer = analysis.Analyzer(print_conf_mat=True, plot=False)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            '''
            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)
            '''
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            #if i % self.print_interval == 0:
                #ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                #self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        pos_precision, pos_recall, pos_f1 = positive_class_scores(
            metrics_dict['precision'], metrics_dict['recall'], self.analyzer.existing_class_ind
        )
        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']
        self.epoch_metrics['precision'] = pos_precision
        self.epoch_metrics['recall'] = pos_recall
        self.epoch_metrics['f1'] = pos_f1
        self._log_subject_scores(predictions, targets, per_batch['IDs'])

        '''
        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)
        '''
        return self.epoch_metrics, metrics_dict

    def _log_subject_scores(self, predictions, targets, id_batches):
        """Precision and recall of the positive class for each session."""
        dataset = getattr(self.dataloader, 'dataset', None)
        table = getattr(dataset, 'hash_table', None)
        if table is None:
            return
        id_parts = []
        for chunk in id_batches:
            if torch.is_tensor(chunk):
                id_parts.append(chunk.detach().cpu().numpy())
            else:
                id_parts.append(np.asarray(chunk))
        ids = np.concatenate(id_parts).reshape(-1)
        stats = {}
        for idx, pred, target in zip(ids, predictions, targets):
            uid = table[int(idx)][0]
            bucket = stats.setdefault(str(uid), {'tp': 0, 'fp': 0, 'fn': 0})
            pred = int(pred)
            target = int(target)
            if pred == 1 and target == 1:
                bucket['tp'] += 1
            elif pred == 1 and target == 0:
                bucket['fp'] += 1
            elif pred == 0 and target == 1:
                bucket['fn'] += 1
        parts = []
        for uid in sorted(stats):
            tp = stats[uid]['tp']
            fp = stats[uid]['fp']
            fn = stats[uid]['fn']
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            parts.append('{} precision {:.3f} recall {:.3f} tp {} fp {} fn {}'.format(
                uid, precision, recall, tp, fp, fn
            ))
        if parts:
            logger.info('Subject scores: %s', ' | '.join(parts))


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    # logger.info("Evaluating on validation set ...")
    # eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)
    # eval_runtime = time.time() - eval_start_time
    # logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    # global val_times
    # val_times["total_time"] += eval_runtime
    # val_times["count"] += 1
    # avg_val_time = val_times["total_time"] / val_times["count"]
    # avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    # avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    # logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    # logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    # logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        best_name = config.get('best_checkpoint_name', 'model_best.pth')
        utils.save_model(os.path.join(config['save_dir'], best_name), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        # pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def train_runner(config, model, trainer, val_evaluator, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])
    # Higher-is-better metrics have to start below any real score. 1e16 only works for loss.
    if config['key_metric'] in NEG_METRICS:
        best_value = 1e16
    else:
        best_value = -1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    best_f1 = -1.0
    checkpoint_interval = config.get('checkpoint_interval', 50)
    save_dir = os.path.dirname(path)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        # Validation loss is minimized by predicting "no transition" on every window.
        # Keep the checkpoint with the best transition-class F1 instead.
        f1 = aggr_metrics_val['f1']
        if f1 > best_f1:
            best_f1 = f1
            utils.save_model(path, epoch, model, optimizer)
            logger.info('Saved best positive-class F1 {:.4f} at epoch {} to {}'.format(f1, epoch, path))
        if checkpoint_interval and epoch % checkpoint_interval == 0:
            prefix = config.get('checkpoint_prefix', 'AEF_model')
            epoch_path = os.path.join(save_dir, '{}_epoch_{}.pth'.format(prefix, epoch))
            utils.save_model(epoch_path, epoch, model, optimizer)
            logger.info('Saved epoch checkpoint {}'.format(epoch_path))

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return