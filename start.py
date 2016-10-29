import numpy as np
import pandas as pd
import cPickle as pkl
from sklearn.cross_validation import train_test_split
with open('./dataset/train_data_pickle','rb') as f1:
    trainf = pkl.load(f1);
trainf = pd.DataFrame.as_matrix(trainf);
#labels are in the last col
