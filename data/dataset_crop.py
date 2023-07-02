# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


class CropDataset(Dataset):

    def __init__(self, dataset, usage='train'):
        super(CropDataset, self).__init__()
        self.dataset = dataset
        self.usage = usage
        self._prepare_data()

    def __len__(self):
        return self.feature_array.shape[0]

    def __getitem__(self, item):
        feature = self.feature_array[item]
        label = self.label_array[item]
        return torch.from_numpy(feature), torch.as_tensor(label).long()

    def _prepare_data(self):
        """
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        :return:
        """
        # read data
        if self.usage == 'train':
            data_file = os.path.join('./datasets', self.dataset, self.dataset + "_train.csv")
        elif self.usage == 'test':
            data_file = os.path.join('./datasets', self.dataset, self.dataset + "_test.csv")

        data_df = pd.read_csv(data_file, sep=',', header=None)
        data_array = np.array(data_df)

        # deal feature and label
        label_array = (data_array[:, 0] + 3).astype(np.int)
        feature_array = data_array[:, 1:].astype(np.float32)

        num_band, time_step = 2, 40
        # reorganize in [batch, channel, time]
        feature_array = feature_array.reshape(-1, num_band, time_step)
        # append vv-vh feature
        diff_feature = feature_array[:, 0, :] - feature_array[:, 1, :]
        diff_feature = diff_feature[:, np.newaxis, :]
        feature_array = np.concatenate((feature_array, diff_feature), axis=1)

        # mean = np.nanmean(train_feature)
        # std = np.nanstd(train_feature)
        # train_feature = (train_feature - mean) / std
        # test_feature = (test_feature - mean) / std

        # warrper data
        self.feature_array = feature_array
        self.label_array = label_array


if __name__ == "__main__":
    crop_dataset = CropDataset(dataset='dijon_8m_4mean')
