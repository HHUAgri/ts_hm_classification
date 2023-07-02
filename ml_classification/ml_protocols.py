# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import class_weight


def fit_lr(features, values, max_samples=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=0, stratify=values)
        features, values = split[0], split[2]

    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=1000000, multi_class='ovr'))
    pipe.fit(features, values)
    return pipe


def fit_knn(features, values):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    pipe.fit(features, values)
    return pipe


def fit_svm(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]
    svm = SVC(C=np.inf, gamma='scale')

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_svm = svm.fit(features, values)
    else:
        grid_search = GridSearchCV(
            svm,
            {
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )

        grid_search.fit(features, values)
        fitted_svm =  grid_search.best_estimator_
    # if

    return fitted_svm


def fit_rf(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]
    rf = RandomForestClassifier(oob_score=True, verbose=1, random_state=10)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_rf = rf.fit(features, values)
    else:
        grid_search = GridSearchCV(
            rf,
            {
                'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [int(x) for x in np.linspace(10, 30, num=5)],
                'criterion': ['gini', 'entropy']
            },
            cv=5, n_jobs=5
        )
        grid_search.fit(features, values)
        fitted_rf = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_rf


def fit_xgb(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]
    xgb = XGBClassifier(n_estimators=20, max_depth=4, min_child_weight=14, subsample=0.5, colsample_bytree=0.5)

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    # sample weight
    sample_weight = None
    # sample_weight = class_weight.compute_sample_weight("balanced", values)
    # sample_weight = np.sqrt(sample_weight)
    # sample_weight = np.power(sample_weight, 2)

    if not cv_search:
        fitted_xgb = xgb.fit(features, values, sample_weight=sample_weight)
    else:
        parameters = {
            # 'n_estimators': range(80, 200, 20),             # 决策树的个数
            # 'max_depth': range(3, 10, 1),                   # 树的最大深度，也是用来避免过拟合的
            # 'min_child_weight': range(1, 5, 1),             # 值越大，越容易欠拟合；值越小，越容易过拟合
            # 'subsample': np.linspace(0.7, 0.9, 10),         # 每棵树随机采样的比例。 减小，算法会保守，避免过拟合。过小，会导致欠拟合。0.5-1
            # 'colsample_bytree': np.linspace(0.5, 0.98, 10), # 用来控制每棵随机采样的列数的占比(列是特征)。0.5-1
            # 'learning_rate': np.linspace(0.1, 0.5, 10),     # 学习率，控制每次迭代更新权重时的步长，0.3

            'n_estimators': range(50, 200, 40),  # 决策树的个数
            'max_depth': range(3, 10, 1),  # 树的最大深度，也是用来避免过拟合的
            'min_child_weight': range(1, 5, 1),  # 值越大，越容易欠拟合；值越小，越容易过拟合
            'subsample': np.linspace(0.7, 0.9, 5),  # 每棵树随机采样的比例。 减小，算法会保守，避免过拟合。过小，会导致欠拟合。0.5-1
            'colsample_bytree': np.linspace(0.5, 0.98, 5),  # 用来控制每棵随机采样的列数的占比(列是特征)。0.5-1
            'learning_rate': np.linspace(0.1, 0.5, 5),     # 学习率，控制每次迭代更新权重时的步长，0.3
        }
        grid_search = GridSearchCV(xgb, parameters, verbose=3, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_xgb = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_xgb


def fit_lgbm(features, values, max_samples=99999, cv_search=False):
    nb_classes = np.unique(values, return_counts=True)[1].shape[0]
    train_size = features.shape[0]
    lgbm = LGBMClassifier()

    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_size > max_samples:
        split = train_test_split(features, values, train_size=max_samples, random_state=100, stratify=values)
        features, values = split[0], split[2]

    if not cv_search:
        fitted_lgbm = lgbm.fit(features, values)
    else:
        parameters = {
            'n_estimators': range(80, 200, 4),
            'max_depth': range(2, 15, 1),
            'learning_rate': np.linspace(0.01, 2, 20),
            'subsample': np.linspace(0.7, 0.9, 20),
            'colsample_bytree': np.linspace(0.5, 0.98, 10),
            'min_child_weight': range(1, 9, 1)
        }
        grid_search = GridSearchCV(lgbm, parameters, cv=5, n_jobs=5)
        grid_search.fit(features, values)
        fitted_lgbm = grid_search.best_estimator_
        print(grid_search.best_params_)
    # if

    return fitted_lgbm
