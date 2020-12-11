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
dataset = pickle.load(open("/content/drive/MyDrive/Szakdolgozat/Data/dataset", "rb" ))

#Creating array for storage of internal values
predictions_asd = []
sign = []
grad = []
classifiers = {}

transability = []

forTest = [84]

#Importing classifiers by ids
for index in dataset.keys():
    path = "/content/drive/MyDrive/Szakdolgozat/Classifiers/classifier_id_{}". format(index)
    name = "classifier{}". format(index)
    name = pickle.load(open(path, 'rb'))
    classifiers['{}' .format(index)] = name

with open("/content/drive/MyDrive/Szakdolgozat/SVM_RACE/classifier", 'rb') as pickle_file:
    model_SVM = pickle.load(pickle_file)

# Load the tensorflow model
model = keras.models.load_model('/content/drive/MyDrive/Szakdolgozat/Neural_Network_Race/DNN_race')
model.trainable = False

#Defining loss object
loss_object = tf.keras.losses.CategoricalCrossentropy()

#Defining gradient calculator modell
def create_adversarial_pattern(input_embedding, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_embedding)
        prediction = model(input_embedding)
        predictions_asd.append(prediction.numpy())
        loss = loss_object(input_label, prediction)
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_embedding)
    grad.append(gradient.numpy())
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    sign.append(signed_grad.numpy())
    return signed_grad

#Choosing datakey (= person)
# ID = 1

#Creating One-Hot encoded input labels
prediction_white = [1, 0, 0, 0]
prediction_black = [0, 1, 0, 0]
prediction_asian = [0, 0, 1, 0]
prediction_indian = [0, 0, 0, 1]

#Declaring epsilon values
epsilons_race = [0.0, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.014]

#Tracking pesron number
person_number = 0

#Creating arrays for data storage
face_embeddings = []
class_error_final = []
class_emb_final = []
class_advex_final = []
prediction_advex_array = []

predSuccessful = {}
classSuccessful = {}

#Creating dict for storage
predSuccessful = {}
for eps in epsilons_race:
    predSuccessful[eps] = []
    
classSuccessful = {}
for eps in epsilons_race:
    classSuccessful[eps] = []

svmResults = {}

for key in dataset.keys():
    svmResults['{}'. format(key)] = {}

#Iterating through persons
for key in dataset.keys():
    
    #Creating arrays for data storage, tracking embedding number
    classifier_emb_array = []
    classifier_advex_emb_array = []
    embedding_number = 0
    
    svmResultsByKey = {}
    for eps in epsilons_race:
        svmResultsByKey['{}'. format(eps)] = []
    
    input_label_variable = [0, 0, 0, 0]
    
    if dataset[key]["race"] == "white":
            input_label_variable = prediction_white
            input_label_string = "white"
    if dataset[key]["race"] == "black":
            input_label_variable = prediction_black
            input_label_string = "black"
    if dataset[key]["race"] == "asian":
            input_label_variable = prediction_asian
            input_label_string = "asian"
    if dataset[key]["race"] == "indian":
            input_label_variable = prediction_indian
            input_label_string = "indian"
            
    
    #Iterating through the chosen person's embeddings
    for emb in dataset[key]["embeddings"]:
        print("\n\n{}. people" .format(person_number))
        #Converting embedding to tensor
        emb = tf.convert_to_tensor([emb])
    
        #Defining input label value due to sex (correct label)
        
        
        input_label = np.asarray(input_label_variable)
        input_label = np.reshape(input_label, (1, 4))
    
        #Creating perturbation
        perturbations = create_adversarial_pattern(emb, input_label)
        print("Embedding number: {}" .format(embedding_number))
        
        #Counting embedding number
        embedding_number = embedding_number + 1
        
        #Iterating through adv. examples with different perturbations
        for eps in epsilons_race:
            #Creating adversarial example
            advex_emb = eps*perturbations + emb
            
            #Predicting sex for advex
            predictedRace = int(model.predict_classes(advex_emb))
            prediction_advex_array.append(predictedRace)
            svmPredict = model_SVM.predict(advex_emb)
            svmResultsByKey["{}". format(eps)].append(svmPredict)
            
            if (predictedRace == 0):
                predictedRace = "white"
            if (predictedRace == 1):
                predictedRace = "black"
            if predictedRace == 2:
                predictedRace = "asian"
            if predictedRace == 3:
                predictedRace = "indian"

            if (svmPredict == 1):
                svmPredict = "white"
            if (svmPredict == 2):
                svmPredict = "black"
            if svmPredict == 3:
                svmPredict = "asian"
            if svmPredict == 4:
                svmPredict = "indian" 
            
            
            print(eps, ": ", "predicted race for emb:", predictedRace, "ground truth:", dataset[key]["race"], "SVM:    ", svmPredict)
            
            if(predictedRace != input_label_string):
                if svmPredict != input_label_string:
                    transability.append(1)
                else:
                    transability.append(0)
            
            #Store face embedding
            face_embeddings.append(advex_emb[0])
            
            #Identification for original emb and advex
            emb_id = int(classifiers["{}" .format(key)].predict(emb))
            classifier_emb_array.append(emb_id)
            advex_id = int(classifiers["{}" .format(key)].predict(advex_emb))
            classifier_advex_emb_array.append(advex_id)
            
            if(predictedRace != input_label_string):
                predSuccessful[eps].append(1)
                            
                if(advex_id == 1):
                    classSuccessful[eps].append(1)
                else:
                    classSuccessful[eps].append(0)
            else:
                predSuccessful[eps].append(0)
    
            print("Classification for emb: ", emb_id, " Classification for advex emb: ", advex_id)
            
    
    #Calculating error of identification due advex       
    classifier_advex_emb_array = np.asarray(classifier_advex_emb_array)
    classifier_emb_array = np.asarray(classifier_emb_array)
    classification_error_advex = (classifier_advex_emb_array[classifier_advex_emb_array == 1].size) / (classifier_emb_array[classifier_emb_array == 1].size)
    class_error_final.append(classification_error_advex)
    
    #Storage of classification
    class_emb_final.append(classifier_emb_array)
    class_advex_final.append(classifier_advex_emb_array)
    
    #Tracking person number
    person_number = person_number + 1

cntForEps = 0

wb = Workbook()
results_excel = wb.add_sheet('Results')
results_excel.write(0, 0, 'Epsilon')
results_excel.write(0, 1, 'Successful classification')
results_excel.write(0, 2, 'Successful identification')
results_excel.write(0, cntForEps+12, 'Epsilon values')
results_excel.write(2, 4, 'SVM')

transability = np.asarray(transability)
SVM_final = (transability[transability == 1].size) / (transability.size)

results_excel.write(3, 4, SVM_final)

for eps in epsilons_race:
    results_excel.write(0, cntForEps+13, epsilons_race[cntForEps])
    cntForEps = cntForEps + 1

lineCounter = 1

# Printing result, for this test not reliable, because divison with 0
for eps in epsilons_race:
    array_ActClass = classSuccessful[eps]
    array_ActClass = np.asarray(array_ActClass)
    print("successfull classicication for eps: ", eps)
    resultOfClassification = array_ActClass[array_ActClass == 1].size / array_ActClass.size
    print(resultOfClassification)
    results_excel.write(lineCounter, 0, eps)
    results_excel.write(lineCounter, 1, resultOfClassification)
    lineCounter = lineCounter  + 1
    
lineCounter = 1
    
for eps in epsilons_race:
    array_ActClass = predSuccessful[eps]
    array_ActClass = np.asarray(array_ActClass)
    print("successfull prediction for eps: ", eps)
    resultOfIdentification = array_ActClass[array_ActClass == 1].size / array_ActClass.size
    print(resultOfIdentification)
    results_excel.write(lineCounter, 2, resultOfIdentification)
    lineCounter = lineCounter + 1

with os.scandir('/content/drive/MyDrive/Szakdolgozat/Results/Final/FGSM/Race/') as entries:
    cnt = 1
    for entry in entries:
        cnt = cnt + 1

nameSave = "/content/drive/MyDrive/Szakdolgozat/Results/Final/FGSM/Race/Results_{}". format(cnt) + ".xls"
wb.save(nameSave)