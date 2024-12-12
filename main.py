
# import torch
# from dataVisuals.dataLoaders import statistic_search, load_initial_data, load_formatted_data, load_change_set_data
# from dataVisuals.test_results import test_results
# from convTran import convTranTrain, convTranTest, convTranTestVisualizer


def start_model():
    from dataset import AEFDataset
    from torch.utils.data import DataLoader
    from aefModel import AefModel
    from config import modelConfig
    from aef_trainer import Trainer
    model_config = modelConfig()
    # Paths for train test sensor label data.
    train_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_change_data_x.csv'
    train_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_change_data_y.csv'
    test_x_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_x.csv'
    test_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/test_change_data_y.csv'
    train_data = AEFDataset(train_x_path, train_y_path, model_config)
    test_data = AEFDataset(test_x_path, test_y_path, model_config)

    # Path to save the model to
    model_config.ckpt_path = '/home/silver/MXTReality/AEF/Models/'

    train_dataloader = DataLoader(train_data, batch_size=model_config.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=32)
    test_dataloader = DataLoader(test_data, batch_size=model_config.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=32)

    model = AefModel(model_config)
    # PATH = '/home/silver/MXTReality/AEF/Models/aefmodel_6.pt'
    # model.load_state_dict(torch.load(PATH))

    trainer = Trainer(model, train_dataloader, test_dataloader, model_config)

    trainer.train()

def start_yolo_model():
    from dataset import YoloDataset
    from YoloModel import YoloModel
    from YoloTrainer import Trainer

    train_path = 'C:/Users/sluco/TERCStudents/ProcessedData'
    test_path = 'C:/Users/sluco/TERCStudents/ProcessValData'
    train_data = YoloDataset(train_path, 'train')
    test_data = YoloDataset(test_path, 'test')

    model = YoloModel()
    trainer = Trainer(model, train_data, test_data, batch_size=32, lr=1e-3)
    trainer.train(epochs=10, save_path="best_model.pth")


def main():
    # start_model()
    # statistic_search()
    # load_initial_data()
    # load_formatted_data()
    # test_results()
    # load_change_set_data()
    # convTranTrain()
    # convTranTest()
    # convTranTestVisualizer()
    start_yolo_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
