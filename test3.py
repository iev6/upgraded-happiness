# coding: utf-8
import numpy as np
import pandas as pd
import cPickle as pkl
get_ipython().magic(u'cd Desktop/lel')
get_ipython().magic(u'ls ')
with open('train_data_pickle','rb') as f1:
    train1 = pkl.load(f1);
    
import xgboost
import xgboost as xgb
labels = train1['label']
labels
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder();
enc.fit(labels)
a1 = enc.transform(labels)
a1
enc.fit([0,1,2,3,4,5,6,7,8,9,10,11,12])
a1 = enc.transform(labels);
labels_OH = np.zeros(30000,12);
labels_OH = np.zeros([30000,12]);
labels_OH.shape
enc
enc.n_values=12
enc
enc.fit(np.transpose(labels));
a1 = enc.transfor(np.transpose(labels));
a1 = enc.transform(np.transpose(labels));
a1
for i in xrange(30000):
    labels_OH(i,labels[i])=1;
    
for i in xrange(30000):
    labels_OH[i,labels[i]]=1;
    
labels_OH
labels_OH[0,:]
labels[0]
labels_OH = np.zeros([30000,12]);
for i in xrange(30000):
    labels_OH[i,labels[i]-1]=1;
    
dtrain = xgb.DMatrix(data=train,label=labels_OH);
train1
train = train1[:,:-1];
import pandas as pd
train1 = pd.DataFrame.as_matrix(train1);
train = train1[:,:-1];
dtrain = xgb.DMatrix(data=train,label=labels_OH);
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6
dtrain = xgb.DMatrix(data=train,label=labels);
from sklearn.cross_validation import train_test_split
train,test = train_test_split(train1,test_size=0.2);
get_ipython().magic(u'xdel dtrain')
get_ipython().magic(u'xdel train1')
dtrain = xgb.DMatrix(data=train[:,:-1],label=train[:,-1]);
dtest = xgb.DMatrix(data=test[:,:-1],label=test[:,-1]);
watchlist = [ (dtrain,'train'), (dtest, 'test') ]
num_round = 5
bst = xgb.train(param, dtrain, num_round, watchlist );
param['num_class'] = 12
bst = xgb.train(param, dtrain, num_round, watchlist );
pred = bst.predict( dtest );
test_Y = test[:,-1];
print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
1-0.627667
param['max_depth'] = 16;
param['num_round']=10;
bst = xgb.train(param, xg_train, num_round, watchlist );
bst = xgb.train(param, dtrain, num_round, watchlist );
get_ipython().magic(u'ls ')
