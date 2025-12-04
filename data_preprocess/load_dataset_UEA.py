import os
from scipy.io.arff import loadarff
import numpy as np
from sklearn.utils import Bunch
import torch
from torch_geometric.data import Data
import pandas as pd


def str2value(label_train, label_test):
    class_label = set(np.concatenate([label_train, label_test], 0))
    class_label = list(class_label)
    num_class = len(class_label)

    dict_class = {}
    for i in range(num_class):
        dict_class[class_label[i]] = i

    label_train_value = []
    for train_i in label_train:
        label_train_value.append(dict_class[train_i])
    label_train_value = np.stack(label_train_value, 0)

    label_test_value = []
    for test_i in label_test:
        label_test_value.append(dict_class[test_i])
    label_test_value = np.stack(label_test_value, 0)

    return label_train_value, label_test_value


def _parse_relational_arff(data):
    X_data = np.asarray(data[0])
    num_sample = len(X_data)
    x, y = [], []

    if X_data[0][0].dtype.names is None:
        for i in range(num_sample):
            x_sample = np.asarray([X_data[i][name] for name in X_data[i].dtype.names])
            x.append(x_sample.T)
            y.append(X_data[i][1])
    else:
        for i in range(num_sample):
            x_sample = np.asarray([
                X_data[i][0][name] for name in X_data[i][0].dtype.names
            ])
            x.append(x_sample.T)
            y.append(X_data[i][1])

    x = np.array(x).astype('float64')
    y = np.array(y)

    try:
        y = y.astype('float64').astype('int64')
    except ValueError:
        y = y.astype('str')

    return x, y


"""Load a UEA data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Missing values are represented as NaN's.

"""


def _load_uea_dataset(dataset, path):
    new_path = os.path.join(path, dataset)
    try:
        descripiton_file = [
            file for file in os.listdir(new_path)
            if ('Description.txt' in file
                or dataset + '.txt' in file)
        ][0]
    except IndexError:
        descripiton_file = None

    if descripiton_file is not None:
        try:
            with(open(os.path.join(new_path, descripiton_file), encoding='utf-8')) as f:
                description = f.read()
        except UnicodeDecodeError:
            with(open(os.path.join(new_path, descripiton_file), encoding='ISO-8859-1')) as f:
                description = f.read()
    else:
        description = None

    train_data = loadarff(os.path.join(new_path, dataset + '_TRAIN.arff'))
    x_train, y_train = _parse_relational_arff(train_data)
    test_data = loadarff(os.path.join(new_path, dataset + '_TEST.arff'))
    x_test, y_test = _parse_relational_arff(test_data)

    bunch = Bunch(
        data_train=x_train, target_train=y_train,
        data_test=x_test, target_test=y_test,
        DESCR=description,
        url=("http://www.timeseriesclassification.com/"
             "description.php?Dataset={}".format(dataset))
    )

    train_data = bunch.data_train
    test_data = bunch.data_test
    train_target = bunch.target_train
    test_target = bunch.target_test

    train_target, test_target = str2value(train_target, test_target)

    return train_data, test_data, train_target, test_target


def load_uea_ts_dataset(filename, path, istrain):
    if istrain:
        path = os.path.join(path, filename, filename + "_eq_TRAIN.ts")
    else:
        path = os.path.join(path, filename, filename + "_eq_TEST.ts")

    df = pd.read_csv(path, sep='\t')
    num_row = df.shape[0]

    label_data = []
    for line in range(num_row):
        line_data = df.loc[line].values[0]
        label_data.append(line_data)
        if line_data == '@data':
            label_data = []

    data = []
    label = []
    for data_i in range(len(label_data)):
        data_split = label_data[data_i].split(':')
        num_sensors = len(data_split) - 1
        signal_sensors = []
        for sensor_i in range(num_sensors):
            sensor_signal_i = []
            for i in data_split[sensor_i].split(','):
                sensor_signal_i.append(float(i))
            sensor_signal_i = np.stack(sensor_signal_i)
            signal_sensors.append(sensor_signal_i)
        label_signal = float(data_split[-1])
        signal_sensors = np.stack(signal_sensors, 0)
        data.append(signal_sensors)
        label.append(label_signal)
    data = np.stack(data, 0)
    label = np.stack(label, 0)

    # data = data[:,:,1::8]

    return data, label


def data_normalizaiton(data):
    num_sensors = data.shape[1]
    data_nor = []
    for sensor_i in range(num_sensors):
        data_i = data[:, sensor_i, :]
        sensor_mean = np.mean(data_i)
        sensor_std = np.std(data_i)
        if sensor_std != 0:
            data_nor.append((data_i - sensor_mean) / (sensor_std))

    data_nor = np.stack(data_nor, 1)

    return data_nor


def data_loader(files_name, root):
    if not os.path.exists(root):
        os.mkdir(root)
    if files_name in arff_read_UEA:
        data_train, data_test, label_train, label_test = _load_uea_dataset(files_name, root)
    else:
        data_train, label_train = load_uea_ts_dataset(files_name, root, True)
        data_test, label_test = load_uea_ts_dataset(files_name, root, False)
        label_train, label_test = str2value(label_train, label_test)

    data_train = data_normalizaiton(data_train)
    data_test = data_normalizaiton(data_test)

    root_saved_path = '../data'
    if not os.path.exists(root_saved_path):
        os.mkdir(root_saved_path)
    if not os.path.exists(os.path.join(root_saved_path, files_name)):
        os.mkdir(os.path.join(root_saved_path, files_name))

    data_train = torch.from_numpy(data_train)
    label_train = torch.from_numpy(label_train)
    data_test = torch.from_numpy(data_test)
    label_test = torch.from_numpy(label_test)

    if data_train.size(-1) % 2 != 0:
        data_train = data_train[:, :, :-1]
        data_test = data_test[:, :, :-1]

    torch.save({'samples': data_train, 'labels': label_train}, os.path.join(root_saved_path, files_name, 'train.pt'))
    torch.save({'samples': data_test, 'labels': label_test}, os.path.join(root_saved_path, files_name, 'test.pt'))


arff_read_UEA = ['ArticularyWordRecognition', 'FingerMovements', 'FaceDetection', 'MotorImagery', 'SelfRegulationSCP1']
ts_read_UEA = ['SpokenArabicDigits', 'InsectWingbeat', 'CharacterTrajectories']


if __name__ == '__main__':
    data_loader("ArticularyWordRecognition", "../TS_Dataset")