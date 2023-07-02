# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import compute_sample_weight

from data_util import load_train_data
import sk_protocols


def performance_metric(y_true, y_predict, y_onehot, y_proba):

    # avg = 'micro'
    # avg = 'macro'
    avg = "weighted"
    # avg = 'samples'
    # sw = compute_sample_weight(class_weight='balanced', y=y_true)

    # accuracy
    acc = accuracy_score(y_true, y_predict)
    # confusion matrix
    cm = confusion_matrix(y_true, y_predict)
    print(cm)
    f1 = f1_score(y_true, y_predict, average=avg)
    precision = precision_score(y_true, y_predict, average=avg)
    recall = recall_score(y_true, y_predict, average=avg)
    mcc = matthews_corrcoef(y_true, y_predict)

    #
    ap = average_precision_score(y_onehot, y_proba, average=avg)
    roc_auc = roc_auc_score(y_onehot, y_proba, average=avg)

    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'mcc': mcc, 'ap': ap, 'roc_auc': roc_auc}


def construct_model(train_data, train_labels, test_data, test_labels, eval_protocol='rocket'):
    print(f"### Classification using {eval_protocol}")

    assert train_labels.ndim == 1
    if eval_protocol == 'rocket':
        fit_clf = sk_protocols.fit_rocket
    elif eval_protocol == 'fcn':
        fit_clf = sk_protocols.fit_fcn
    elif eval_protocol == 'lstmfcn':
        fit_clf = sk_protocols.fit_lstmfcn
    elif eval_protocol == 'tapnet':
        fit_clf = sk_protocols.fit_rocket
    elif eval_protocol == 'inceptiontime':
        fit_clf = sk_protocols.fit_inceptiontime
    else:
        assert False, 'unknown evaluation protocol'
    trained_model = fit_clf(train_data, train_labels, cv_search=False)

    pred_labels = trained_model.predict(train_data)
    pred_proba = trained_model.predict_proba(train_data)
    labels_onehot = label_binarize(train_labels, classes=np.arange(train_labels.max() + 1))
    train_scores = performance_metric(train_labels, pred_labels, labels_onehot, pred_proba)
    print('\033[34m### Training performance: {}\033[0m'.format(train_scores))

    # test_acc = trained_model.score(test_x, test_y)
    # print('### Overall accuracy is {} for testing'.format(test_acc))
    pred_labels = trained_model.predict(test_data)
    pred_proba = trained_model.predict_proba(test_data)
    labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
    test_scores = performance_metric(test_labels, pred_labels, labels_onehot, pred_proba)
    print('\033[34m### Testing performance: {}\033[0m'.format(test_scores))

    print('### Construct model complete')
    return trained_model


def classification_train():
    """
    时序分类模型训练
    """

    dataset = 'dijon_s1_mean'
    eval_protocol = 'inceptiontime'

    # read training datasets
    sample_feature, sample_label = load_train_data(dataset)

    # label encode
    le = LabelEncoder()
    sample_label = le.fit_transform(sample_label)

    # split train and test samples
    train_feature, test_feature, train_label, test_label = train_test_split(sample_feature, sample_label, test_size=0.2, shuffle=True)

    # training and saving classifier
    model = construct_model(train_feature, train_label, test_feature, test_label, eval_protocol=eval_protocol)

    # saving everything
    # model, label_encoder

    # clean and return
    return model


def main():
    print("##########################################################")
    print("###  #####################################################")
    print("##########################################################")

    classification_train()

    print("### Complete! #############################################")


if __name__ == "__main__":
    main()
