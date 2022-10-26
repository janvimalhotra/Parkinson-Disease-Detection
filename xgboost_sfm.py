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
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(xgb.XGBClassifier(criterion='entropy'))
sel.fit(X_train,Y_train)
X_Train=pd.DataFrame(X_train)
X_Test=pd.DataFrame(X_test)
selected_feat= X_Train.columns[(sel.get_support())]
len(selected_feat)
rde=X_Train.columns
rfe=X_Test.columns
for col in rde:
    if col not in selected_feat:
      del X_Train[col]
for col in rfe:
    if col not in selected_feat:
      del X_Test[col]
      ni1 = xgb.XGBClassifier(criterion = 'entropy')
ni1.fit(X_Train, Y_train)
X_Train_prediction=ni1.predict(X_Train)
Training_data_accuracy =accuracy_score(Y_train, X_Train_prediction)
print(Training_data_accuracy)
X_Test_prediction=ni1.predict(X_Test)
Test_data_accuracy =accuracy_score(Y_test, X_Test_prediction)
print(Test_data_accuracy)
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
X_Test_prediction=ni1.predict(X_Test)
Test_data_precision =precision_score(Y_test, X_Test_prediction)
print(Test_data_precision)
X_Test_prediction=ni1.predict(X_Test)
Test_data_precision =precision_score(Y_test, X_Test_prediction)
print(Test_data_precision)
X_Test_prediction=ni1.predict(X_Test)
Test_data_recall =recall_score(Y_test, X_Test_prediction)
print(Test_data_recall)
X_Test_prediction=ni1.predict(X_Test)
Test_data_f1 =f1_score(Y_test, X_Test_prediction)
print(Test_data_f1)
X_Test_prediction=ni1.predict(X_Test)
Test_data_roc_auc=roc_auc_score(Y_test, X_Test_prediction)
print(Test_data_roc_auc)
plot_confusion_matrix(ni1,X_Test,Y_test)
from sklearn import metrics
metrics.plot_roc_curve(ni1,X_Test,Y_test)