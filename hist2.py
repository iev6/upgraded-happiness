
run hist1.py
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,criterion='entropy', min_samples_split=5)
import csv
arr=[]
with open('test_data','rb') as f:
    rea=csv.reader(f)
    for r in rea:
        arr.append(map(float,r))
test=np.array(arr)
rf_p=rf.predict_proba(c)
cf_p=clf.predict_proba(c)
from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=[('rf',rf),('etf',clf)],voting='soft')
for cl in [rf,clf,vc]:
    scores=cross_val_score(cl,c,y)
    print scores.mean()
