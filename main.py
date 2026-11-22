import argparse

from preprocessing.dataLoaders import load_change_set_data, load_gaze_change_set_data
from ConvTran.train import convTranTrain
from ConvTran.export_test_predictions import export_test_predictions


def main(gaze=False):
    if gaze:
        print('Preprocessing HeadsetFinal sessions with gaze')
        load_gaze_change_set_data()
    else:
        print('Preprocessing HeadsetFinal sessions')
        load_change_set_data()
    print('Training ConvTran')
    convTranTrain(gaze=gaze)
    print('Exporting test predictions')
    export_test_predictions(gaze=gaze)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess HeadsetFinal sessions and train ConvTran')
    parser.add_argument('--gaze', action='store_true', help='Include gaze channels and write gaze-named outputs')
    args = parser.parse_args()
    main(gaze=args.gaze)
