import cPickle as pickle
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.cross_validation import cross_val_score

with open('train_data_pickle','rb') as f:
    a=pickle.load(f).as_matrix()
with open('ipca_500comp.pkl','rb') as f:
    b=pickle.load(f)

x=a[:,:-1]
y=a[:,-1]
c=b.transform(x)

clf = ExtraTreesClassifier(n_estimators=500, max_depth=None,min_samples_split=2, random_state=0)
# cross-val = 40%

from sklearn import svm
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
#
# k_fold=KFold(3)
# scores=[mo.fit(c[train], y[train]).score(c[test], y[test]) for train, test in k_fold.split(c)]
