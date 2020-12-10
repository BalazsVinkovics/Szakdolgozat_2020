# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:47:05 2020

@author: Balázs Vinkovics
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import xlwt
from xlwt import Workbook
import os

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/10_11/Data/dataset", "rb" ))

#Creating array for storage of internal values
predictions_asd = []
sign = []
grad = []
classifiers = {}

#Importing demographic data classifiers
race_model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks2/DNN_race')
sex_model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks_sex/DNN_race')
age_model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks3/DNN_race')

#Importing classifiers by ids
for index in dataset.keys():
    path = "D:/Szakdolgozat/10_11/Data/Classifiers/classifier_id_{}". format(index)
    name = "classifier{}". format(index)
    name = pickle.load(open(path, 'rb'))
    classifiers['{}' .format(index)] = name

iterationNumber_sex = 50
iterationNumber_race = 50
iterationNumber_age = 50

universalEmbeddingKey_race = []
universalEmbeddingKey_age = []
universalEmbeddingKey_sex = []

targets_age = []
targets_sex = []
targets_race = []

predSucRace = []
predSucAge = []
predSucSex = []

idSuc = []
keys_stored = []

#Defining loss object
loss_object = tf.keras.losses.CategoricalCrossentropy()

#Defining gradient calculator modell
def create_adversarial_pattern_race(input_embedding, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_embedding)
        prediction = race_model(input_embedding)
        loss = loss_object(input_label, prediction)
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_embedding)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(-gradient)
    return signed_grad

def create_adversarial_pattern_sex(input_embedding, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_embedding)
        prediction = sex_model(input_embedding)
        loss = loss_object(input_label, prediction)
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_embedding)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(-gradient)
    return signed_grad

def create_adversarial_pattern_age(input_embedding, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_embedding)
        prediction = age_model(input_embedding)
        loss = loss_object(input_label, prediction)
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_embedding)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(-gradient)
    return signed_grad

#Choosing datakey (= person)
forTest = [1325, 84]

prediction_white = [1, 0, 0, 0]
prediction_black = [0, 1, 0, 0]
prediction_asian = [0, 0, 1, 0]
prediction_indian = [0, 0, 0, 1]

prediction_female = [0.0, 1.0]
prediction_male = [1.0, 0.0]

prediction_0_20 = [1.0, 0.0, 0.0, 0.0]
prediction_20_40 = [0.0, 1.0, 0.0, 0.0]
prediction_40_60 = [0.0, 0.0, 1.0, 0.0]
prediction_60_80 = [0.0, 0.0, 0.0, 1.0]

#Declaring epsilon values
epsilons_sex = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005]
epsilons_race = [0.006, 0.0062, 0.0064, 0.0066, 0.0068, 0.007, 0.0072, 0.0074]
epsilons_age = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.0014]

#Tracking pesron number
person_number = 0

#Creating arrays for data storage
face_embeddings = []
class_error_final = []
class_emb_final = []
class_advex_final = []
prediction_advex_array = []

universalEmbedding = []
embForKeys = []

predRace = []
predSex = []
predAge = []

#Iterating through persons
for key in dataset.keys():
    
    embForEmbs = []
    keys_stored.append(key)
    
    #Creating arrays for data storage, tracking embedding number
    classifier_emb_array = []
    classifier_advex_emb_array = []
    embedding_number = 0
    
    universalEmbeddings_race = []
    universalEmbeddings_sex = []
    universalEmbeddings_age = []
    
    input_label_variable_race = [0, 0, 0, 0]
    input_label_variable_age = [0, 0, 0, 0]
    input_label_variable_sex = [0, 0]
    
    age_min = int(dataset[key]["age"].split(",")[0])
    age_max = int(dataset[key]["age"].split(",")[1])
    
    if dataset[key]["race"] == "white":
            input_label_variable_race = prediction_asian
            target_label_string_race = "asian"
    if dataset[key]["race"] == "black":
            input_label_variable_race = prediction_indian
            target_label_string_race = "indian"
    if dataset[key]["race"] == "asian":
            input_label_variable_race = prediction_black
            target_label_string_race = "black"
    if dataset[key]["race"] == "indian":
            input_label_variable_race = prediction_white
            target_label_string_race = "white"
            
    if dataset[key]["sex"] == "m":
            input_label_variable_sex = prediction_female
            target_label_string_sex = "f"
    if dataset[key]["sex"] == "f":
            input_label_variable_sex = prediction_male
            target_label_string_sex = "m"  
    
    if 0 <= age_min:
        if age_max <= 20:
            input_label_variable_age = prediction_20_40
            target_label_string_age = "20_40"
            dataset[key]["age"] = "0_20"
    if 20 <= age_min:
        if age_max <= 40:
            input_label_variable_age = prediction_40_60
            target_label_string_age = "40_60"
            dataset[key]["age"] = "20_40"
    if 40 <= age_min:
        if age_max <= 60:
            input_label_variable_age = prediction_60_80
            target_label_string_age = "60_80"
            dataset[key]["age"] = "40_60"
    if 60 <= age_min:
        if age_max <= 80:
            input_label_variable_age = prediction_0_20
            target_label_string_age = "0_20"
            dataset[key]["age"] = "60_80"
    
    universalEmbeddingOne = []
        
    for i in range (len((dataset[key]["embeddings"][0]))):        
        universalEmbeddingOne.append(0)      
    universalEmbeddingOne = np.asarray(universalEmbeddingOne)
    universalEmbeddingOne = universalEmbeddingOne.reshape(128,1).T
        
    for embs in dataset[key]["embeddings"]:
        embs = np.asarray(embs)
        embs = embs.reshape(128, 1).T
        universalEmbeddingOne = universalEmbeddingOne + embs
    universalEmbeddingOne = universalEmbeddingOne / float(len(dataset[key]["embeddings"]))
    dataset[key]["embeddings"] = universalEmbeddingOne
    
#Iterating through persons
for key in dataset.keys():
    
    emb = dataset[key]["embeddings"][0]
    
    emb = np.asarray(emb)
    emb = emb.reshape(128, 1).T
    
    race = race_model.predict_classes(emb)
    sex = sex_model.predict_classes(emb)
    age = age_model.predict_classes(emb)
    
    if (race == 0):
        race = "white"
    if (race == 1):
        race = "black"
    if race == 2:
        race = "asian"
    if race == 3:
        race = "indian"
    
    if (sex == 0):
        sex = "m"
    if (sex == 1):
        sex = "f"  
    
    if (age == 0):
        age = "0_20"
    if (age == 1):
        age = "20_40"
    if age == 2:
        age = "40_60"
    if age == 3:
        age = "60_80"
    
    print("prediction: ", race, "ground_truth: ", dataset[key]["race"])
    if race == dataset[key]["race"]:
        predRace.append(1)
    else:
        predRace.append(0)
        
    print("prediction: ", age, "ground_truth: ", dataset[key]["age"])
    if age == dataset[key]["age"]:
        predAge.append(1)
    else:
        predAge.append(0)
        
    print("prediction: ", sex, "ground_truth: ", dataset[key]["sex"])
    if sex == dataset[key]["sex"]:
        predSex.append(1)
    else:
        predSex.append(0)
        
predSex = np.asarray(predSex)
predAge = np.asarray(predAge)
predRace = np.asarray(predRace)

predSexRes = predSex[predSex == 1].size / predSex.size
predAgeRes = predAge[predAge == 1].size / predAge.size
predRaceRes = predRace[predRace == 1].size / predRace.size

checkSex = []

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/10_11/Data/dataset", "rb" ))

for key in dataset.keys():
    for emb in dataset[key]["embeddings"]:
        emb = np.asarray(emb)
        emb = emb.reshape(128, 1).T
        
        
        sex = sex_model.predict(emb)
        print(sex)
        sex = sex_model.predict_classes(emb)
        print(sex)
        
        if (sex == 0):
            sex = "m"
        if (sex == 1):
            sex = "f"  
        
        print("pediciton: ", sex, "ground truth: ", dataset[key]["sex"])
        
        if sex == dataset[key]["sex"]:
            checkSex.append(1)
        else:
            checkSex.append(0)
        
checkSex = np.asarray(checkSex)
value = checkSex[checkSex == 1].size / checkSex.size

print(value)