# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tree_hierarchy_label_crop_source = [
    [1, 21],
    [2, 21],
    [3, 21],
    [4, 21],
    [5, 22],
    [6, 23],
    [7, 23],
    [8, 23],
    [9, 24],
    [10, 24],
    [11, 24],
    [12, 24]
]
tree_hierarchy_label_crop = [
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 1],
    [9, 2],
    [10, 2],
    [11, 2],
    [12, 3],
    [13, 3],
    [14, 3],
    [15, 3]
]
tree_total_nodes_crop = 16
tree_levels_crop = 2


def find_hierarchy_label(leaf_label):
    return_value = None
    for it in tree_hierarchy_label_crop:
        if it[0] == leaf_label:
            return_value = it
    if return_value is None:
        print("Could not find hierarchy label in tree")
    return return_value


def get_hierarchy_label(leaf_label, dataset='crop'):

    upper_label_list = []
    target_list_sig = []

    for i in range(leaf_label.size(0)):
        if dataset == 'crop':
            hierarchy_label = find_hierarchy_label(int(leaf_label[i]))
            upper_label_list.append(hierarchy_label[1])

        target_list_sig.append(int(leaf_label[i]))
    # for

    upper_label_list = torch.from_numpy(np.array(upper_label_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    return upper_label_list, target_list_sig
