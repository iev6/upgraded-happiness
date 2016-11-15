import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.grid_search import GridSearchCV

import pandas as pd

with open('train_data_pickle', mode='rb') as f:
    trainf = pkl.load(f)
trainf = pd.DataFrame.as_matrix(trainf)
x = trainf[:,:-1]
y = trainf[:,-1]

print 'data loaded'

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'multi:softmax'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                            cv_params,
                             scoring = 'accuracy', cv = 5, n_jobs = 4)
# Disabling grid search
# optimized_GBM.fit(x, y)
xgdmat = xgb.DMatrix(x, y)
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'multi:softmax', 'max_depth':3, 'min_child_weight':1, 'num_class':12}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)
