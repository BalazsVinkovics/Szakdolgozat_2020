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

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/dataframe_kenez/dataframe_balanced_dlib.pickle", "rb" ))

embeddings_all = []
array= []

#Declaring expected predicitons for races
prediction_white = [1.0, 0.0, 0.0, 0.0]
prediction_black = [0.0, 1.0, 0.0, 0.0]
prediction_asian = [0.0, 0.0, 1.0, 0.0]
prediction_indian = [0.0, 0.0, 0.0, 1.0]

for emb in dataset.iloc():
    
    emb = emb.tolist()
    emb = list(np.float_(emb))
        
    #Setting expected label
    if emb[130] == 0:
        emb[130] = (prediction_white)
    if emb[130] == 1:
        emb[130] = (prediction_black)
    if emb[130] == 2:
        emb[130] = (prediction_asian)
    if emb[130] == 3:
        emb[130] = (prediction_indian)
                
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

X_val_emb = []
Y_val_label = []

# X_train_emb, X_val_emb = train_test_split(X_train_emb, train_size = 0.8, test_size = 0.2)
# Y_train, Y_val_label = train_test_split(Y_test_label, train_size = 0.8, test_size = 0.2)

#Creating training and test feature and label set
for emb in training_dataset:
    X_train_emb.append(emb[0:128])
X_train_emb = np.asarray(X_train_emb)

for emb in test_dataset:
    X_test_emb.append(emb[0:128])
X_test_emb = np.asarray(X_test_emb)
    
for emb in training_dataset:
    Y_train_label.append(emb[130])
Y_train_label = np.asarray(Y_train_label)
    
for emb in test_dataset:
    Y_test_label.append(emb[130])
Y_test_label = np.asarray(Y_test_label)
    

#Creating model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(128)),
    keras.layers.Dense(96, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation="softmax")
])

#Training model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
history = model.fit(X_train_emb, Y_train_label, epochs=100, validation_split=0.2, steps_per_epoch=2)
test_loss, test_acc = model.evaluate(X_test_emb, Y_test_label, verbose=2)

#Checking result, saving model
prediction_check = np.asarray(X_test_emb)
result = model.predict(prediction_check)
model.save("D:/Szakdolgozat/Neural_Networks2_v2/DNN_race")

#Plot results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('A rassz modell pontossága')
plt.ylabel('pontosság [%]')
plt.xlabel('epoch [-]')
plt.legend(['tanítási érték', 'tesztelési érték'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('A rassz modell hibája')
plt.ylabel('hiba [-]')
plt.xlabel('epoch [-]')
plt.legend(['tanítási érték', 'tesztelési érték'], loc='upper right')
plt.show()

# label_to_predict = []
# counter = 0
# numberOfRace = 3
# truthRace = ""
# arrayForPrediction = []

# #Checking for asian people the prediciton, getting asian keys
# for array in Y_test_label:
#     if numberOfRace == 0:
#         truthRace = "white"
#     if numberOfRace == 1:
#         truthRace = "black"
#     if numberOfRace == 2:
#         truthRace = "asian"
#     if numberOfRace == 3:
#         truthRace = "indian"
        
#     if array[numberOfRace] == 1:
#         label_to_predict.append(counter)
#     counter = counter + 1
# #Creating predictions, printing results
# for number in label_to_predict:
#     pred = model.predict(X_test_emb[number].reshape(1,128))
    
#     max = np.max(pred)
#     prediction_string = "Error"
#     if (pred[0][0] == max):
#         prediction_string = "white"
#         arrayForPrediction.append("white")
#     if (pred[0][1] == max):
#         prediction_string = "black"
#         arrayForPrediction.append("black")
#     if (pred[0][2] == max):
#         prediction_string = "asian"
#         arrayForPrediction.append("asian")
#     if (pred[0][3] == max):
#         prediction_string = "indian"
#         arrayForPrediction.append("indian")
            
#     print(prediction_string)
# arrayForPrediction = np.asarray(arrayForPrediction)
# print(arrayForPrediction[arrayForPrediction == truthRace].size / arrayForPrediction.size)