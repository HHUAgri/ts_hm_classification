# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np
import pandas as pd


def load_train_data(dataset='dijon'):
    print(f"### loading data from {dataset}")

    # read data
    train_file = os.path.join('../datasets', dataset, dataset + "_train.csv")
    train_df = pd.read_csv(train_file, sep=',', header=None)
    train_array = np.array(train_df)

    # deal train arrays
    train_label = train_array[:, 0].astype(np.int8)
    train_feature = train_array[:, 1:].astype(np.float32)

    return train_feature, train_label,


def load_predict_data(dataset='dijon'):

    # print(f"### loading data from {dataset}")
    #
    # # read data
    # train_file = os.path.join('../datasets/CROP', dataset, dataset + "_train.csv")
    # test_file = os.path.join('../datasets/CROP', dataset, dataset + "_test.csv")
    #
    # train_df = pd.read_csv(train_file, sep=',', header=None)
    # test_df = pd.read_csv(test_file, sep=',', header=None)
    # train_array = np.array(train_df)
    # test_array = np.array(test_df)
    #
    # # Move the labels to {0, ..., L-1}
    # labels = np.unique(train_array[:, 0])
    # transform = {k: i for i, k in enumerate(labels)}
    # train_label = np.vectorize(transform.get)(train_array[:, 0])
    # test_label = np.vectorize(transform.get)(test_array[:, 0])
    #
    # # deal train and test arrays
    # train_feature = train_array[:, 1:].astype(np.float32)
    # test_feature = test_array[:, 1:].astype(np.float32)
    # assert (train_feature.shape[-1] == test_feature.shape[-1])
    #
    # return train_feature, train_label, test_feature, test_label

    pass
