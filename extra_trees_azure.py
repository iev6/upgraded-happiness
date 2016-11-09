# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.cross_validation import cross_val_score

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print('Input pandas.DataFrame #1:\r\n\r\n{0}'.format(dataframe1))

    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    X=dataframe1.as_matrix()
    y=dataframe2.as_matrix().ravel()
    print X.shape,y.shape
    print('working')
    #clf = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0, n_jobs=10)
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)    
    scores = cross_val_score(clf, X, y)
    print(scores.mean())
    # df=pd.DataFrame(data=mm,index=1+np.arange(len(mm)),columns=1+np.arange(len(mm[0])))
    
    # Return value must be of a sequence of pandas.DataFrame
    return dataframe1
