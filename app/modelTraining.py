
#Script to obtain data 
# from helpers import *
import numpy as np 
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#Libraries to create the multiclass model
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
#Import tensorflow and disable the v2 behavior and eager mode
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

#Library to validate the model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data_moodsnew.csv")


le = LabelEncoder()

col_features = df.columns[6:-3]
ms = MinMaxScaler().fit(df[col_features])
X = ms.transform(df[col_features])
X2 = np.array(df[col_features])
Y = df['mood']

#Encodethe categories
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)


#Convert to  dummy (Not necessary in my case)
# dummy_y = np_utils.to_categorical(encoded_y)
# dummy_y = np_utils.to_categorical(encoded_y)

X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15, stratify = encoded_y)

target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)

# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# acc_svcT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
# acc_svc, acc_svcT
#
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# acc_knnT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
# acc_knn, acc_knnT
#
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# acc_gaussianT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
# acc_gaussian, acc_gaussianT
#
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# acc_linear_svcT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
# acc_linear_svc, acc_linear_svcT
#
# sgd = SGDClassifier()
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# acc_sgdT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
# acc_sgd, acc_sgdT
#
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# acc_decision_tree, acc_decision_treeT
# acc_decision_treeT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forestT = sklearn.metrics.accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
acc_random_forest, acc_random_forestT

# models = pd.DataFrame({
#     'Model': ['Support Vector Machines', 'KNN',
#               'Random Forest', 'Naive Bayes',
#               'Stochastic Gradient Decent', 'Linear SVC',
#               'Decision Tree'],
#     'Score': [acc_svc, acc_knn,
#               acc_random_forest, acc_gaussian,
#               acc_sgd, acc_linear_svc, acc_decision_tree]})
# models.sort_values(by='Score', ascending=False)

# models = pd.DataFrame({
#     'Model': ['Support Vector Machines', 'KNN',
#               'Random Forest', 'Naive Bayes',
#               'Stochastic Gradient Decent', 'Linear SVC',
#               'Decision Tree'],
#     'Score': [acc_svcT, acc_knnT,
#               acc_random_forestT, acc_gaussianT,
#               acc_sgdT, acc_linear_svcT, acc_decision_treeT]})
# models.sort_values(by='Score', ascending=False)

random_forest.predict([X_test[70]])

rDict = {
"danceability": 0.61,
"energy": 0.582,
"key": 0,
"loudness": -10.422,
"mode": 1,
"speechiness": 0.044,
"acousticness": 0.119,
"instrumentalness": 0.0000376,
"liveness": 0.12,
"valence": 0.781,
"tempo": 82.492,
"type": "audio_features",
"id": "1ppuHX1oVMku5LTL0swNZP",
"uri": "spotify:track:1ppuHX1oVMku5LTL0swNZP",
"track_href": "https://api.spotify.com/v1/tracks/1ppuHX1oVMku5LTL0swNZP",
"analysis_url": "https://api.spotify.com/v1/audio-analysis/1ppuHX1oVMku5LTL0swNZP",
"duration_ms": 251333,
"time_signature": 4
}

fList = ['duration_ms', 'danceability', 'acousticness', 'energy', 'instrumentalness',
       'liveness', 'valence', 'loudness', 'speechiness', 'tempo']

OList = []
for key in fList:
  OList.append(rDict[key])

inp = [194680, 0.58, 0.867, 0.421, 0, 0.205, 0.347, -9.161, 0.0315, 114.725]
inp = [317427, 0.457, 0.000239, 0.904, 0.0879, 0.396, 0.48, -5.303, 0.0747, 141.038]
inp = OList

# inp = df[col_features].iloc[43,:].to_list()
inpT = ms.transform([inp])
random_forest.predict(inpT)
pred = encoder.classes_[random_forest.predict(inpT)[0]]
print(pred)
"""### Predictions"""
filename = 'random_forest.sav'
pickle.dump(random_forest, open(filename, 'wb'))
np.save('classes.npy', encoder.classes_)
scaler_filename = "scaler.save"
joblib.dump(ms, scaler_filename)

# And now to load...



# some time later...

# load the model from disk
# pickle.loads()
# col_features

# inp = df[col_features].iloc[43,:].to_list()
# inpT = ms.transform([inp])
# random_forest.predict(inpT)
# encoder.classes_[random_forest.predict(inpT)[0]]

