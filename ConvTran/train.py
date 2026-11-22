import os

import torch
from torch.utils.data import DataLoader
from preprocessing.config import change_data_path
from .config import modelConfig, convTranConfig, apply_run_paths
from .dataset import AEFConvDataset
from .Models.optimizers import get_optimizer
from .Models.loss import get_loss_module
from .Models.model import model_factory
from .Models.utils import load_model
from .trainer import SupervisedTrainer, train_runner
from .event_score import score_distance_peaks


def convTranTrain(gaze=False):
    print('Event precision from 3-second motion peaks')
    score_distance_peaks(gaze=gaze)

    model_config = modelConfig()
    train_x_path = change_data_path('train', 'x', gaze=gaze)
    train_y_path = change_data_path('train', 'y', gaze=gaze)
    test_x_path = change_data_path('test', 'x', gaze=gaze)
    test_y_path = change_data_path('test', 'y', gaze=gaze)
    # Fixed 240-frame (8 s) blocks.
    train_data = AEFConvDataset(train_x_path, train_y_path, model_config, augment=True)
    test_data = AEFConvDataset(test_x_path, test_y_path, model_config, augment=True)

    train_dataloader = DataLoader(train_data, batch_size=model_config.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=28)
    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=False,
                                 pin_memory=True, num_workers=28)

    config = apply_run_paths(convTranConfig, gaze=gaze)
    n_features = int(train_data.model_input.shape[1] - 1)
    config['Data_shape'] = [config['batch_size'], n_features, model_config.max_timestep]

    model = model_factory(config)

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], config['last_checkpoint_name'])
    model.to(config['device'])

    trainer = SupervisedTrainer(model, train_dataloader, config['device'], config['loss_module'], config['optimizer'],
                                l2_reg=0, print_interval=config['print_interval'], console=config['console'],
                                print_conf_mat=False)

    val_evaluator = SupervisedTrainer(model, test_dataloader, config['device'], config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=False)

    train_runner(config, model, trainer, val_evaluator, save_path)

    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_model, test_dataloader, config['device'], config['loss_module'],
                                            print_interval=config['print_interval'], console=config['console'],
                                            print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)
