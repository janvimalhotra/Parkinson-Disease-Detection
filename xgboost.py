import pandas as pd
parkinson_data=pd.read_csv('/content/pd_speech_features.csv')
#importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
pd_data=parkinson_data.drop(columns=['id'],axis=1)
pd_data=parkinson_data.drop(columns=['id'],axis=1)
pd_data['first']=pd_data['mean_MFCC_2nd_coef']
min, max = pd_data['first'].quantile([0.1,0.99])
pf1=pd_data[(pd_data['first']>min)&(pd_data['first']<max)]
#
pf1['second']=pf1['tqwt_kurtosisValue_dec_26']
min1, max1 = pf1['second'].quantile([0.1,0.99])
pf2=pf1[(pf1['second']>min1)&(pf1['second']<max1)]
#
pf2['third']=pf2['tqwt_kurtosisValue_dec_36']
min2, max2 = pf2['third'].quantile([0.1,0.99])
parkinsons_data=pf2[(pf2['third']>min2)&(pf2['third']<max2)]
#

#creating X and Y
X=parkinsons_data.drop(columns=['class'],axis=1)
Y=parkinsons_data['class']
#Spliting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
ni = xgb.XGBClassifier()
ni.fit(X_train, Y_train)

X_train_prediction=ni.predict(X_train)
training_data_accuracy =accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)
X_test_prediction=ni.predict(X_test)
test_data_accuracy =accuracy_score(Y_test, X_test_prediction)
print(test_data_accuracy)
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
X_test_prediction=ni.predict(X_test)
test_data_precision =precision_score(Y_test, X_test_prediction)
print(test_data_precision)
X_test_prediction=ni.predict(X_test)
test_data_recall =recall_score(Y_test, X_test_prediction)
print(test_data_recall)
X_test_prediction=ni.predict(X_test)
test_data_f1 =f1_score(Y_test, X_test_prediction)
print(test_data_f1)
plot_confusion_matrix(ni,X_test,Y_test)
metrics.plot_roc_curve(ni,X_test,Y_test)
from sklearn.metrics import roc_auc_score
X_test_prediction=ni.predict(X_test)
test_data_roc=roc_auc_score(Y_test, X_test_prediction)
print(test_data_roc)