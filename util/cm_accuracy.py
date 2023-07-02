# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix


def confusion_matrix_accuracy(cm_array):

    label_count = cm_array.shape[0]

    gt_list = []
    pred_list = []
    for rr in range(label_count):
        for cc in range(label_count):
            rc_count = cm_array[rr][cc]
            gt_list.extend([rr] * rc_count)
            pred_list.extend([cc] * rc_count)
        # for
    # for
    gt_array = np.array(gt_list)
    pred_array = np.array(pred_list)

    acc = accuracy_score(gt_array, pred_array)
    f1 = f1_score(gt_array, pred_array, average='weighted', zero_division=0)
    precision = precision_score(gt_array, pred_array, average='weighted', zero_division=0)
    recall = recall_score(gt_array, pred_array, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(gt_array, pred_array)

    print('\nacc = %.4f, f1 = %.4f, p = %.4f, r = %.4f, mcc = %.4f.\n' % (acc, f1, precision, recall, mcc))
    print(confusion_matrix(gt_array, pred_array))
    print(classification_report(gt_array, pred_array))

    pass


def main():

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

    hmc_summer_cm = [
        [178, 66, 2],
        [60, 190, 8],
        [14, 11, 56]
    ]

    resnet_summer_cm = [
        [243, 66, 1],
        [182, 114, 1],
        [69, 30, 7]
    ]

    cm = np.array(hmc_summer_cm, dtype=np.int)

    confusion_matrix_accuracy(cm)

    pass


if __name__ == "__main__":
    main()
