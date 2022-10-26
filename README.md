# Parkinson-Disease-Detection
There are 3 machiine learning models for Parkinson's disease detection based on extracted voice features.
Dataset has been downloaded from kaggle orignally available on uci machine learning repository. Link of data set:
https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features
Xgboost and Extra Tress Classifier are algorithms choosen by me.
OUtliers are removed
Feature selection technique(entropy based) is applied in files xgboost_sfm.py and et_sfm.py while in xgboost.py the model has been created without any feature selection techinque.
It was observed that Extra Trees Classifier with feature selection yields best result.
