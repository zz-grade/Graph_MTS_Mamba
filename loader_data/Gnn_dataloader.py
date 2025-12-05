import os

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from .augmentations import *


class Load_Dataset(Dataset):
    def __init__(self, dataset, configs, args):
        super(Load_Dataset, self).__init__()

        x_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(x_train.shape) < 3:
            x_train = x_train.unsqueeze(2)
        if x_train.shape.index(min(x_train)) != 1:
            x_train = x_train.permute(0, 2, 1)
        if isinstance(x_train, np.ndarray):
            self.x_data = torch.from_numpy(x_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = x_train.float()
            self.y_data = y_train.long()

        self.len = len(self.x_data)
        shape = self.x_data.size()
        self.x_data = torch.reshape(self.x_data, (shape[0], shape[1], configs.time_denpen_len, configs.window_size))
        self.x_data = torch.transpose(self.x_data, 1, 2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



class Load_Training_Data(Dataset):
    def __init__(self, dataset, configs, args, train=True):
        super().__init__()
        self.args = args
        self.configs = configs
        x_train = dataset['samples']
        y_train = dataset['labels']


        if configs.wavelet_aug:
            x_train_aug1 = x_train.numpy()
            x_train_aug2 = x_train.numpy()
            x_train_aug1 = wavelet_transform(x_train_aug1, True)
            x_train_aug2 = wavelet_transform(x_train_aug2, False)
            x_train_aug1 = torch.from_numpy(x_train_aug1)
            x_train_aug2 = torch.from_numpy(x_train_aug2)
        else:
            x_train_aug1 = x_train
            x_train_aug2 = x_train

        if isinstance(x_train, np.ndarray):
            self.x_data = torch.from_numpy(x_train)
            self.x_data_aug1 = torch.from_numpy(x_train_aug1)
            self.x_data_aug2 = torch.from_numpy(x_train_aug2)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = x_train.float()
            self.x_data_aug1 = x_train_aug1.float()
            self.x_data_aug2 = x_train_aug2.float()
            self.y_data = y_train.long()

        if self.x_data.shape[0] != self.y_data.shape[0]:
            print("x_data len:", self.x_data.shape[0])
            print("y_data len:", self.y_data.shape[0])
            raise ValueError("samples 和 labels 数量不一致！")
        self.len = self.x_data.shape[0]
        shape = self.x_data.size()
        # if train:
        #     time_denpen_len = configs.window_size_train
        #     window_size = configs.time_denpen_len_train
        # else:
        #     time_denpen_len = configs.window_size_val
        #     window_size = configs.time_denpen_len_val
        self.x_data = self.x_data.reshape(shape[0], shape[1], configs.time_denpen_len, configs.window_size)
        self.x_data = torch.transpose(self.x_data, 1, 2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def augmentation(self, input, weak):
        bs, time_length, num_nodes, feature_dimension = input.size()
        input = torch.reshape(input, [bs, num_nodes * time_length, feature_dimension])
        input = DataTransform(input, self.configs)
        if weak:
            input = input[0]
        else:
            input = input[1]
        input = np.array(input)
        input = torch.from_numpy(input).float()
        input = torch.reshape(input, [bs, time_length, num_nodes, feature_dimension])

        return input


def data_generator(data_path, configs, args):
    train_dataset = torch.load(os.path.join(data_path, 'train.pt'))
    val_dataset = torch.load(os.path.join(data_path, 'val.pt'))
    test_dataset = torch.load(os.path.join(data_path, 'test.pt'))

    train_dataset = Load_Training_Data(train_dataset, configs, args)
    val_dataset = Load_Training_Data(val_dataset, configs, args, False)
    test_dataset = Load_Training_Data(test_dataset, configs, args)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, drop_last=configs.drop_last, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, drop_last=configs.drop_last, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader, test_loader
