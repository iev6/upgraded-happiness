import cPickle as pickle
import numpy as np
with open('train_data_pickle','rb') as f:
        a=pickle.load(f).as_matrix()
x=a[:,:-1]
y=a[:,-1]
from sklearn.decomposition import PCA
pca = PCA(n_components=1024)
pca.fit(x)
with open('pca_1024.pkl','wb') as f:
    pickle.dump(pca,f)
