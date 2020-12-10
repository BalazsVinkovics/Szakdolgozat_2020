# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:19:31 2020

@author: User
"""

import pickle
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/dataframe_kenez/dataframe_balanced_dlib.pickle", "rb" ))


embeddings_all = []
array= []

#Declaring expected predicitons for races
prediction_0_20 = 1
prediction_20_40 = 2
prediction_40_60 = 3
prediction_60_80 = 4

for emb in dataset.iloc():
    
        flag = 1
    
        emb = emb.tolist()
        emb = list(np.float_(emb))
        
        
        if(flag == 1):
            if 0 <= emb[128]:
                if emb[128] <= 20:
                    emb[128] = prediction_0_20
                    flag = 0
        if(flag == 1):
            if 21 <= emb[128]:
                if emb[128] <= 40:
                    emb[128] = prediction_20_40
                    flag = 0
        if(flag == 1):
            if 41 <= emb[128]:
                if emb[128] <= 60:
                    emb[128] = prediction_40_60
                    flag = 0
        if(flag == 1):
            if 61 <= emb[128]:
                if emb[128] <= 80:
                    emb[128] = prediction_60_80
                    flag = 0
                
        embeddings_all.append(emb)

# embeddings_all = np.asarray(embeddings_all)
training_dataset, test_dataset = train_test_split(embeddings_all, train_size=0.8, test_size=0.2)
random.shuffle(training_dataset)

X_train_emb = []
X_test_emb = []
Y_train_label = []
Y_test_label = []

X_train_normal = []
X_test_normal = []
X_outliers = []

#Creating training and test feature and label set
for emb in training_dataset:
    X_train_emb.append(emb[0:128])
X_train_emb = np.asarray(X_train_emb)

for emb in test_dataset:
    X_test_emb.append(emb[0:128])
X_test_emb = np.asarray(X_test_emb)
    
for emb in training_dataset:
    Y_train_label.append(emb[128])
Y_train_label = np.asarray(Y_train_label)
    
for emb in test_dataset:
    Y_test_label.append(emb[128])
Y_test_label = np.asarray(Y_test_label)
    
#Creating One Class Classifiers
clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
clf.fit(X_train_emb, Y_train_label)
y_pred_train = clf.predict(X_test_emb)

result = []
cnt = 0

for label in y_pred_train:
    if label == Y_test_label[cnt]:
        result.append(1)
    else:
        result.append(0)
    cnt = cnt + 1
    
result = np.asarray(result)
finalResult = result[result == 1].size / result.size

# destination = ("D:/Szakdolgozat/SVM_AGE/classifier")
# pickle.dump(clf, open(destination, "wb" ))