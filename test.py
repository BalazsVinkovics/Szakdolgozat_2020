# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:26:47 2020

@author: User
"""
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


asd_asd = []

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/10_11/Data/dataset", "rb" ))
# Load the tensorflow model
model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks2/DNN_race')
model.trainable = False

for _ in dataset.keys():
    asd = []
    if dataset[_]["race"] == "black":
        embeddings = dataset[_]["embeddings"]
        asd.append(dataset[_]["race"])
        print("\n\ntruth: ", dataset[_]["race"])
        for __ in range(50):
            emb = np.asarray(embeddings[__])
            emb = emb.reshape(1,128)
            prediction = model.predict(emb)
            
            max = np.max(prediction)
            prediction_string = "Error"
            if (prediction[0][0] == max):
                prediction_string = "white"
            if (prediction[0][1] == max):
                prediction_string = "black"
            if (prediction[0][2] == max):
                prediction_string = "asian"
            if (prediction[0][3] == max):
                prediction_string = "indian"
            asd.append(prediction_string)
            print(prediction_string)
            asd_asd.append(asd)
        