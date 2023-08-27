# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# np.random.seed(42)


def cm_guide_map(cm_array, map_df, field_name='TYPE'):

    # normalize confusion matrix
    class_lens = len(cm_array)
    cm_array = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]

    # add a column to pandas dataframe
    feature_lens = len(map_df.index)
    new_col = np.zeros(feature_lens, dtype=int)
    map_df = map_df.assign(new_col=new_col)

    # select rows based on condition from pandas dataframe
    relabel_list = []
    for cc in range(class_lens):
        # select rows based on condition from pandas dataframe
        label_df = map_df.loc[map_df['ZHOU_CODE'] == (cc + 1)]
        label_lens = len(label_df)
        # calculate ratios fro splitting columns
        label_indices = (cm_array[cc] * label_lens).astype(int)
        label_diff = label_lens - label_indices.sum()
        label_indices[cc] = label_indices[cc] + label_diff
        #
        assert label_indices.sum() == label_lens
        label_indices = np.cumsum(label_indices)[:-1]

        # split according to rations
        shuffled = label_df.sample(frac=1)
        split_list = np.array_split(shuffled, label_indices)
        # relabel their types
        for ii, part in enumerate(split_list):
            # if ii == cc: continue
            part['new_col'] = (ii+1)
        # merge sub-parts
        relabel_df = pd.concat(split_list)
        relabel_list.append(relabel_df)
    # for

    # reconstruct map dataframe
    relabel_map = pd.concat(relabel_list)
    relabel_map = relabel_map.sort_index()
    relabel_map.rename(columns={'new_col': field_name}, inplace=True)

    # return
    # gt_array = np.array(relabel_map['ZHOU_CODE'])
    # pt_array = np.array(relabel_map[field_name])
    # print(confusion_matrix(gt_array, pt_array))
    return relabel_map


def main():
    # confusion matrix
    hmc_cm = [
        [1568, 146, 25, 24, 25, 37, 50, 2, 6, 24, 54, 14],
        [80, 721, 32, 25, 52, 19, 32, 1, 1, 5, 21, 1],
        [17, 44, 635, 1, 8, 3, 15, 4, 0, 7, 10, 0],
        [29, 40, 3, 72, 4, 5, 4, 0, 2, 4, 12, 0],
        [44, 85, 23, 9, 162, 6, 24, 7, 0, 10, 21, 5],
        [84, 53, 15, 8, 24, 178, 66, 2, 4, 5, 24, 2],
        [96, 55, 34, 0, 32, 60, 190, 8, 0, 3, 10, 3],
        [15, 19, 18, 3, 21, 14, 11, 56, 1, 3, 2, 0],
        [5, 3, 3, 0, 0, 1, 1, 0, 994, 3, 40, 0],
        [28, 25, 17, 3, 13, 3, 4, 0, 5, 146, 90, 14],
        [34, 25, 9, 13, 9, 10, 8, 0, 28, 35, 3329, 77],
        [15, 7, 0, 0, 7, 0, 2, 0, 7, 8, 110, 142]
    ]
    tapnet_cm = [
        [1522, 176, 50, 0, 5, 15, 7, 0, 42, 3, 145, 0],
        [100, 738, 47, 1, 2, 4, 11, 0, 2, 1, 66, 0],
        [15, 63, 563, 0, 2, 1, 4, 0, 7, 0, 26, 0],
        [42, 61, 18, 2, 1, 0, 0, 0, 5, 0, 37, 0],
        [65, 207, 57, 0, 43, 1, 17, 0, 10, 3, 44, 1],
        [154, 96, 36, 2, 4, 53, 67, 0, 19, 0, 65, 0],
        [134, 108, 82, 0, 5, 25, 74, 1, 11, 0, 39, 0],
        [33, 45, 53, 0, 9, 5, 15, 2, 0, 0, 16, 0],
        [5, 0, 2, 0, 0, 0, 1, 0, 944, 0, 65, 0],
        [38, 38, 28, 0, 2, 0, 4, 0, 10, 37, 177, 0],
        [47, 25, 11, 1, 1, 1, 1, 0, 52, 9, 3491, 1],
        [22, 9, 8, 0, 0, 1, 3, 0, 16, 5, 219, 21]
    ]
    resnet_cm = [
        [1570, 169, 16, 6, 10, 106, 30, 0, 4, 17, 71, 23],
        [95, 695, 24, 13, 15, 56, 54, 0, 2, 0, 34, 17],
        [18, 44, 549, 0, 1, 24, 39, 1, 0, 5, 9, 2],
        [48, 58, 3, 21, 0, 34, 5, 0, 2, 0, 27, 2],
        [57, 115, 20, 4, 62, 58, 37, 4, 4, 8, 15, 12],
        [65, 50, 8, 0, 2, 243, 66, 1, 2, 1, 28, 9],
        [104, 57, 18, 0, 4, 182, 114, 1, 1, 0, 10, 6],
        [19, 31, 25, 4, 10, 69, 30, 7, 1, 5, 4, 2],
        [6, 0, 0, 0, 0, 0, 0, 0, 985, 0, 23, 14],
        [28, 32, 13, 2, 6, 26, 6, 3, 2, 90, 114, 38],
        [18, 12, 2, 2, 3, 30, 2, 0, 35, 13, 3219, 170],
        [8, 8, 1, 0, 2, 9, 2, 0, 5, 6, 104, 147]
    ]
    lstm_cm = [
        [1500, 219, 52, 15, 24, 57, 16, 0, 10, 14, 81, 2],
        [78, 764, 34, 17, 35, 24, 15, 0, 1, 1, 23, 0],
        [12, 101, 569, 1, 5, 8, 6, 0, 0, 0, 11, 0],
        [34, 80, 3, 25, 3, 3, 2, 1, 3, 2, 26, 0],
        [39, 167, 38, 7, 93, 12, 11, 0, 2, 5, 19, 0],
        [110, 99, 25, 11, 13, 180, 31, 1, 4, 1, 35, 0],
        [125, 101, 49, 1, 33, 123, 60, 0, 1, 3, 18, 1],
        [18, 43, 26, 3, 17, 22, 17, 2, 3, 0, 7, 0],
        [10, 2, 3, 0, 0, 0, 0, 0, 978, 0, 58, 0],
        [31, 41, 16, 4, 15, 3, 4, 0, 8, 86, 152, 3],
        [42, 39, 10, 8, 8, 9, 1, 0, 28, 20, 3329, 5],
        [20, 13, 4, 1, 2, 5, 3, 0, 8, 9, 195, 30]
    ]
    xgb_cm = [
        [1510, 183, 44, 19, 21, 41, 42, 5, 27, 19, 100, 2],
        [87, 731, 39, 15, 25, 12, 36, 1, 3, 10, 46, 0],
        [26, 58, 563, 1, 4, 1, 20, 0, 5, 2, 19, 0],
        [30, 70, 0, 29, 5, 0, 3, 0, 1, 1, 28, 2],
        [51, 182, 37, 1, 70, 15, 30, 3, 3, 9, 36, 1],
        [120, 73, 35, 8, 14, 100, 88, 1, 10, 4, 34, 1],
        [109, 102, 43, 0, 16, 55, 119, 5, 7, 1, 29, 1],
        [20, 34, 27, 0, 14, 16, 22, 6, 0, 5, 3, 0],
        [7, 1, 1, 0, 0, 1, 2, 0, 983, 1, 76, 0],
        [38, 28, 11, 2, 9, 2, 4, 3, 7, 79, 155, 3],
        [42, 24, 10, 6, 6, 1, 4, 0, 46, 18, 3372, 14],
        [24, 7, 3, 0, 2, 2, 1, 3, 11, 9, 192, 24]
    ]
    cm_array = np.array(xgb_cm, dtype=int)

    # using pandas to read csv files
    map_csv_path = r'K:\FF\application_dataset\2020-france-agri-hmc\parcel-result\parcel_utm.csv'
    map_df = pd.read_csv(map_csv_path, header=0)

    # relabel map
    relabel_map = cm_guide_map(cm_array, map_df, field_name='XGB_TYPE')
    
    # save pandas dataframe into csv file
    relabel_map_path = r'K:\FF\application_dataset\2020-france-agri-hmc\parcel-result\parcel_utm_xgb.csv'
    relabel_map.to_csv(relabel_map_path, index=False)

    print("### Task over")


if __name__ == "__main__":
    main()
