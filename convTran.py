import os

import numpy as np
import torch
from dataset import AEFConvDataset as AEFDataset, AEFTestConvDataset, AEFInferenceConv
from torch.utils.data import DataLoader
from config import modelConfig, convTranConfig
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.model import model_factory
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from torch.utils.tensorboard import SummaryWriter
from dataVisuals.visualizer import switch_visualizer


def convTranTrain():
    model_config = modelConfig()
    train_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_change_data_x.csv'
    train_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_change_data_y.csv'
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_y.csv'
    train_data = AEFDataset(train_x_path, train_y_path, model_config)
    test_data = AEFDataset(test_x_path, test_y_path, model_config)

    train_dataloader = DataLoader(train_data, batch_size=model_config.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=28)
    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=28)

    config = convTranConfig

    model = model_factory(config)

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], 'AEF_' + 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
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


def convTranTest():
    model_config = modelConfig()
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_y.csv'
    test_data = AEFTestConvDataset(test_x_path, test_y_path, model_config)

    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=28)

    config = convTranConfig

    model = model_factory(config)

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], 'AEF_' + 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(config['device'])

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
    print(f"all_metrics {all_metrics}")


def convTranTestVisualizer():
    model_config = modelConfig()
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_y.csv'
    test_data = AEFTestConvDataset(test_x_path, test_y_path, model_config)

    # With shuffle off, I should be getting a number of frames in a row.
    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=False, num_workers=28)

    config = convTranConfig

    model = model_factory(config)

    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], 'AEF_' + 'model_{}.pth'.format('last'))
    model.to(config['device'])

    best_model = load_model(model, save_path)
    best_model.to(config['device'])

    switch_visualizer(model, test_dataloader, model_config)


def convTranTestWOLabel():
    model_config = modelConfig()
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_x.csv'
    test_data = AEFInferenceConv(test_x_path, "Unused String", model_config)

    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=False,
                                 pin_memory=True, num_workers=28)

    config = convTranConfig

    model = model_factory(config)

    save_path = os.path.join(config['save_dir'], 'AEF_' + 'model_{}.pth'.format('last'))
    model.to(config['device'])

    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(config['device'])

    predictions_array = []
    for i, batch in enumerate(test_dataloader):
        X, _, _ = batch
        predictions = model(X.to(config.device))
        predictions_array.append(torch.argmax(predictions).detach())

    predictions_array = np.asarray(predictions_array)
    np.save("this is a save path", predictions_array)
