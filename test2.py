<<<<<<< HEAD
# coding: utf-8
import numpy as np
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd Downloads')
get_ipython().magic(u'ls ')
get_ipython().set_next_input(u'train = np.loads');get_ipython().magic(u'pinfo np.loads')
train = np.loads('train_data_pickle')
import cPickle as pkl
get_ipython().set_next_input(u'train = pkl.loads');get_ipython().magic(u'pinfo pkl.loads')
get_ipython().set_next_input(u'train = pkl.load');get_ipython().magic(u'pinfo pkl.load')
train = pkl.loads('train_data_pickle');
file1 = open('train_data_pickle','rb');
train = pkl.load(file1);
train.shape
train_mean = np.mean(train)
np.linalg.norm(train_mean)
train_base = train[:,:-1];
train[0,:]
train[0,1]
type(train)
train.columns
labels = train['label']
np.histogram(labels)
get_ipython().magic(u'pinfo np.histogram')
np.histogram(labels,12)
get_ipython().magic(u'pinfo np.historgram')
get_ipython().magic(u'pinfo np.histogram')
range(0,12)
np.histogram(labels,range(0,12))
np.histogram(labels,range(0,13))
import pandas as pd
pd.value_counts(labels)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=4);
get_ipython().magic(u'pinfo pd.factorize')
features = train.columns[:-1];
len(features)
clf.fit(train[features],labels);
clf
clf.score(train[features],labels)
clf.feature_importances_
import sklearn
get_ipython().magic(u'pinfo sklearn.model_selection.train_test_split')
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
get_ipython().magic(u'pinfo StratifiedKFold')
x = train[features];
y = labels;
skf = StratifiedKFold(y,n_folds = 12);
skf
skf[1]
skf.n_folds
for train_index, test_index in skf:
    X_train,X_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf.fit(X_train,y_train)
    score1 = clf.score(X_test,y_test);
    print str(score1)+str('\n')
    
train_mat = train.as_matrix()l
train_mat = train.as_matrix();
train_mat.shape
x = train_mat[:,:-1];
y = train_mat[:,-1];
y
skf = StratifiedKFold(y,n_folds=12)
for train_index, test_index in skf:
    X_train,X_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf.fit(X_train,y_train)
    score1 = clf.score(X_test,y_test);
    print str(score1)+str('\n')
    
    
get_ipython().magic(u'save 1-61 test2.py')
=======
# coding: utf-8
import numpy as np
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ..')
get_ipython().magic(u'cd Downloads')
get_ipython().magic(u'ls ')
get_ipython().set_next_input(u'train = np.loads');get_ipython().magic(u'pinfo np.loads')
train = np.loads('train_data_pickle')
import cPickle as pkl
get_ipython().set_next_input(u'train = pkl.loads');get_ipython().magic(u'pinfo pkl.loads')
get_ipython().set_next_input(u'train = pkl.load');get_ipython().magic(u'pinfo pkl.load')
train = pkl.loads('train_data_pickle');
file1 = open('train_data_pickle','rb');
train = pkl.load(file1);
train.shape
train_mean = np.mean(train)
np.linalg.norm(train_mean)
train_base = train[:,:-1];
train[0,:]
train[0,1]
type(train)
train.columns
labels = train['label']
np.histogram(labels)
get_ipython().magic(u'pinfo np.histogram')
np.histogram(labels,12)
get_ipython().magic(u'pinfo np.historgram')
get_ipython().magic(u'pinfo np.histogram')
range(0,12)
np.histogram(labels,range(0,12))
np.histogram(labels,range(0,13))
import pandas as pd
pd.value_counts(labels)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=4);
get_ipython().magic(u'pinfo pd.factorize')
features = train.columns[:-1];
len(features)
clf.fit(train[features],labels);
clf
clf.score(train[features],labels)
clf.feature_importances_
import sklearn
get_ipython().magic(u'pinfo sklearn.model_selection.train_test_split')
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
get_ipython().magic(u'pinfo StratifiedKFold')
x = train[features];
y = labels;
skf = StratifiedKFold(y,n_folds = 12);
skf
skf[1]
skf.n_folds
for train_index, test_index in skf:
    X_train,X_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf.fit(X_train,y_train)
    score1 = clf.score(X_test,y_test);
    print str(score1)+str('\n')
    
train_mat = train.as_matrix()l
train_mat = train.as_matrix();
train_mat.shape
x = train_mat[:,:-1];
y = train_mat[:,-1];
y
skf = StratifiedKFold(y,n_folds=12)
for train_index, test_index in skf:
    X_train,X_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf.fit(X_train,y_train)
    score1 = clf.score(X_test,y_test);
    print str(score1)+str('\n')
    
    
get_ipython().magic(u'save 1-61 test2.py')
>>>>>>> 64c7e9cc4e0218c583787f365933f224bebb0c9e
