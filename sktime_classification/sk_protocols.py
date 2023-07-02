# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import class_weight

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier


def fit_rocket(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    rocket = RocketClassifier(num_kernels=500)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_rocket = rocket.fit(features, values)
    else:
        parameters = {
            # 'num_kernels': range(1000, 1000, 20000),             # 决策树的个数
        }
        grid_search = GridSearchCV(rocket, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_rocket = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_rocket


def fit_fcn(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    fcn = FCNClassifier(n_epochs=99, batch_size=256, verbose=True)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_fcn = fcn.fit(features, values)
    else:
        parameters = {
            # 'num_kernels': range(1000, 1000, 20000),             # 决策树的个数
        }
        grid_search = GridSearchCV(fcn, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_fcn = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_fcn


def fit_lstmfcn(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    lstmfcn = LSTMFCNClassifier(n_epochs=99, batch_size=64, verbose=2)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_lstmfcn = lstmfcn.fit(features, values)
    else:
        parameters = {
            # 'num_kernels': range(1000, 1000, 20000),             # 决策树的个数
        }
        grid_search = GridSearchCV(lstmfcn, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_lstmfcn = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_lstmfcn


def fit_tapnet(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    tapnet = TapNetClassifier(n_epochs=99, batch_size=64, verbose=True)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_tapnet = tapnet.fit(features, values)
    else:
        parameters = {
            # 'num_kernels': range(1000, 1000, 20000),             # 决策树的个数
        }
        grid_search = GridSearchCV(tapnet, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_tapnet = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_tapnet


def fit_inceptiontime(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    inceptiontime = InceptionTimeClassifier(n_epochs=8, batch_size=64, verbose=True)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_inceptiontime = inceptiontime.fit(features, values)
    else:
        parameters = {
            # 'num_kernels': range(1000, 1000, 20000),             # 决策树的个数
        }
        grid_search = GridSearchCV(inceptiontime, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_inceptiontime = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_inceptiontime

