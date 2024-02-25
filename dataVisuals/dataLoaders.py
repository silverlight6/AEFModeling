import pandas as pd
import xgboost as xgb
import glob
import os
from dataVisuals.util import train_test_split
from dataVisuals.visualizer import show_variable_comparison, show_confustion_matrix, analyze_model_features

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


DROP_LIST_1 = [
    'num_sessions',
    'Unnamed: 21',
    'headset',
    'correct_date_time',
    'study_num',
    'timestamp',
    'initial_code',
]

DROP_LIST_2 = [
    'auto_code',
    # 'unique_id'
]

FEATURES = [
    'headset_position_x',
    'headset_position_y',
    'headset_position_z',
    'headset_rotation_w',
    'headset_rotation_x',
    'headset_rotation_y',
    'headset_rotation_z',
    'gaze_origin_x',
    'gaze_origin_y',
    'gaze_origin_z',
    'gaze_direction_x',
    'gaze_direction_y',
    'gaze_direction_z',
]


def load_initial_data():
    path = '/home/silver/MXTReality/AEF/Data/Sensor'
    all_files = glob.glob(os.path.join(path, "*.csv"))

    li = []
    count = 0
    for filename in all_files:
        count += 1
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
        li.append(df)

    sensor_data = pd.concat(li, ignore_index=True)

    sensor_data['unique_id'] = sensor_data['subject'].astype(str) + '_' + \
                               sensor_data['study_num'].astype(str) + '_' + sensor_data['num_sessions'].astype(str)

    label_mapping = {'p': 0, 'n': 1, 'd': 2}
    sensor_data['auto_code'] = sensor_data['auto_code'].replace(label_mapping)
    sensor_data['auto_code'] = sensor_data['auto_code'].replace(to_replace=r'^[^0-9]*$', value=2, regex=True)
    sensor_data['auto_code'] = sensor_data['auto_code'].apply(pd.to_numeric, errors='coerce')

    sensor_data = sensor_data.drop(DROP_LIST_1, axis=1)
    sensor_data.to_csv('/home/silver/MXTReality/AEF/Data/FormattedData/sensor_data.csv', index=False)

    label = sensor_data[['auto_code', 'unique_id']].copy()
    sensor_data = sensor_data.drop(DROP_LIST_2, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(sensor_data, label, 0.9)
    X_train.to_csv('/home/silver/MXTReality/AEF/Data/FormattedData/train_data_x.csv', index=False)
    X_test.to_csv('/home/silver/MXTReality/AEF/Data/FormattedData/test_data_x.csv', index=False)
    y_train.to_csv('/home/silver/MXTReality/AEF/Data/FormattedData/train_data_y.csv', index=False)
    y_test.to_csv('/home/silver/MXTReality/AEF/Data/FormattedData/test_data_y.csv', index=False)

    print(sensor_data.head())
    print(sensor_data.keys())


def load_formatted_data():
    path = '/home/silver/MXTReality/AEF/Data/FormattedData/sensor_data.csv'
    sensor_data = pd.read_csv(path)
    print(sensor_data.keys())

    label = sensor_data[['auto_code', 'unique_id']].copy()
    sensor_data = sensor_data.drop(DROP_LIST_2, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(sensor_data, label, 0.9)

    x_df = pd.DataFrame(data=X_train, columns=FEATURES)
    x_test_df = pd.DataFrame(data=X_test, columns=FEATURES)

    # This line is hella sus. Tried to do it via row, but it was throwing errors for me.
    x_df.fillna(x_df.mean(), inplace=True)
    x_test_df.fillna(x_test_df.mean(), inplace=True)
    x_df['focus'] = y_train['auto_code']
    x_test_df['focus'] = y_test['auto_code']

    x_df = x_df.reset_index(drop=True)
    x_test_df = x_test_df.reset_index(drop=True)

    x = x_df.loc[:, FEATURES].values
    x_test = x_test_df.loc[:, FEATURES].values
    x = StandardScaler().fit_transform(x)
    x_test = StandardScaler().fit_transform(x_test)

    n_comp = 12
    pca_stats = PCA(n_components=n_comp)
    pca_test_stats = PCA(n_components=n_comp)
    principle_components = pca_stats.fit_transform(x)
    principle_test_components = pca_test_stats.fit_transform(x_test)

    principle_stats = pd.DataFrame(data=principle_components, columns=['principle_component_{}'.format(i)
                                                                       for i in range(n_comp)])
    principle_test_stats = pd.DataFrame(data=principle_test_components, columns=['principle_component_{}'.format(i)
                                                                                 for i in range(n_comp)])
    # print(sum(pca_stats.explained_variance_ratio_))
    # print(sum(pca_test_stats.explained_variance_ratio_))
    # print(pca_stats.explained_variance_ratio_)

    model = xgb.XGBClassifier(n_estimators=160,
                              max_depth=6,
                              learning_rate=0.1,
                              subsample=0.5,
                              gamma=2,
                              eta=0.8)

    print(x_df.info)
    print(x_test_df.info)

    train_model = model.fit(principle_stats, x_df.loc[:, 'focus'].values)
    pred = train_model.predict(principle_test_stats)
    cm = confusion_matrix(x_test_df.loc[:, 'focus'].values, pred)

    print('Confusion Matrix')
    print(cm)
    # accuracies = cross_val_score(estimator=model, X=principle_stats, y=x_df.loc[:, 'focus'].values, cv=15)
    # print(f'Mean: {accuracies.mean()}')
    # print(f'Std: {accuracies.std()}')

    analyze_model_features(model, X_train)
    show_confustion_matrix(cm)

    x_df['focus'].replace(0, "productive", inplace=True)
    x_df['focus'].replace(1, "nan", inplace=True)
    x_df['focus'].replace(2, "distracted", inplace=True)
    show_variable_comparison("principle_component_0", "principle_component_1", x_df, principle_stats)

def statistic_search():
    path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_data_x.csv'
    sensor_data = pd.read_csv(path)
    sensor_data.fillna(0, inplace=True)
    sensor_data = sensor_data.drop('unique_id', axis=1)

    sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.max() - sensor_data.min())

    train_y_path = '/home/silver/MXTReality/AEF/Data/FormattedData/train_data_y.csv'
    labels = pd.read_csv(train_y_path)
    labels.fillna(1, inplace=True)

    print(f"max of sensor data {sensor_data.max(axis=0)}")
    print(f"min of sensor data {sensor_data.min(axis=0)}")
    print(f"max of labels {labels.max(axis=0)}")
    print(f"min of labels {labels.min(axis=0)}")
