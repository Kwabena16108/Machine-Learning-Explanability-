#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:34:31 2022

@author: Dixon domfeh
"""

import platform
import sys
import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, cohen_kappa_score, make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve, SCORERS
from sklearn.metrics import log_loss, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from operator import itemgetter
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, truncnorm, norm, randint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklego.preprocessing import ColumnCapper
from xgboost.core import XGBoostError
from sklearn.inspection import permutation_importance


# %% 


class tree_val(object):
    
    def zero_missing_unique(self, df):
        """
        Calculate missing rate, zero rate and unique value of a df
        """
        
        var_miss_rate = df.isnull().sum(axis=0)/df.shape[0]
        var_miss_rate = var_miss_rate.to_frame('missing_rate')
        
        var_zero_rate = (df==0).astype(int).sum(axis=0)/df.shape[0]
        var_zero_rate = var_zero_rate.to_frame('zero_rate')
        
        var_unique = df.nunique().to_frame('count_unique')
        var_type = df.dtypes.to_frame('data_type')
        data_stat = pd.concat([var_miss_rate, var_zero_rate, var_unique, var_type], axis=1)
        
        return data_stat
    
    def GridSearchcv_tune_model(self, model, param_grid, X_train, y_train, *args, **kwargs):
        """
        this function builds a tree based ML model using gridsearchcv with logloss as performance metric
        """
        model_gs = model_selection.GridSearchCV(model, param_grid, scoring="neg_log_loss", return_train_score=True, *args, **kwargs)
        model_gs.fit(X_train, y_train)
        cv_results = pd.DataFrame(model_gs.cv_results_)
        
        return model_gs, model_gs.best_estimator_, cv_results
    
    # define partial dependency for PDPs
    def partial_dependence(self, classifier, X, y, feature_ids = [], f_id = -1):
        X_temp = X.copy()
        grid = np.linspace(np.percentile(X_temp[:, f_id], 0.1),
                           np.percentile(X_temp[:, f_id], 99.5),
                           50)
        y_pred = np.zeros(len(grid))
        for i, val in enumerate(grid):
            X_temp[:, f_id] = val
            y_pred[i] = np.average(classifier.predict_proba(X_temp)[:, 1])
            
        return grid, y_pred


















