# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:20:10 2020

@author: Balázs Vinkovics
"""

#Importing libaries
import pickle
import numpy as np
import pandas as pd
import xlwt
from xlwt import Workbook
import tensorflow as tf
from tensorflow import keras
import os

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/10_11/Data/dataset", "rb" ))

#Creating array for storage of internal values
predictions_asd = []
sign = []
grad = []
classifiers = {}
iterationNumber = 50
loss_stored = []

#Dataset keys for test
forTest = [1364, 1325]

#Importing classifiers by ids
for index in dataset.keys():
    path = "D:/Szakdolgozat/10_11/Data/Classifiers/classifier_id_{}". format(index)
    name = "classifier{}". format(index)
    name = pickle.load(open(path, 'rb'))
    classifiers['{}' .format(index)] = name

# Load the tensorflow model
model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks2/DNN_race')
model.trainable = False

#Defining loss object
loss_object = tf.keras.losses.CategoricalCrossentropy()
# optimizer.lr = 0.01

#Defining gradient calculator modell
def create_adversarial_pattern(input_embedding, target_label):
    with tf.GradientTape() as tape:
        tape.watch(input_embedding)
        prediction = model(input_embedding)
        predictions_asd.append(prediction.numpy())
        loss = loss_object(target_label, prediction)
        loss_stored.append(loss.numpy())
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_embedding)
    grad.append(gradient.numpy())
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(-gradient)
    sign.append(signed_grad.numpy())
    return signed_grad

#Choosing datakey (= person)
# ID = 1

#Creating OneHotEncoding target labels
prediction_white = [1.0, 0.0, 0.0, 0.0]
prediction_black = [0.0, 1.0, 0.0, 0.0]
prediction_asian = [0.0, 0.0, 1.0, 0.0]
prediction_indian = [0.0, 0.0, 0.0, 1.0]

#Setting epsilon values
epsilons_race = [0.0002, 0.00021, 0.00022, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004]

#Tracking person number
person_number = 0

#Creating arrays for data storage
face_embeddings = []
class_error_final = []
class_emb_final = []
class_advex_final = []
prediction_advex_array = []

#Arrays for results of races
array_white_suc = []
array_asian_suc = []
array_black_suc = []
array_indian_suc = []

#Creating dict for storage
predSuccessful = {}
for eps in epsilons_race:
    predSuccessful[eps] = []
    
classSuccessful = {}
for eps in epsilons_race:
    classSuccessful[eps] = []

#Creating array for classification result
#Creating array for prediction results
per_array = []
embs_array = []

#Iterating through persons by keys, for test we use forTest array wit dedicated keys (only 1325 for now)
for key in forTest:
    
    sign.append(key)
    
    #Creating arrays for data storage, tracking embedding number
    classifier_emb_array = []
    classifier_advex_emb_array = []
    
    #Tracking embedding number
    embedding_number = 0
    
    #Creating target label with initial value
    input_label_variable = [0.0, 0.0, 0.0, 0.0]
    
    #Setting target label due to race
    if dataset[key]["race"] == "white":
            input_label_variable = prediction_asian
            target_label_string_new = "asian"
    if dataset[key]["race"] == "black":
            input_label_variable = prediction_indian
            target_label_string_new = "indian"
    if dataset[key]["race"] == "asian":
            input_label_variable = prediction_black
            target_label_string_new = "black"
    if dataset[key]["race"] == "indian":
            input_label_variable = prediction_white
            target_label_string_new = "white"
    
    #Iterating through the chosen person's embeddings
    for emb in dataset[key]["embeddings"]:
        
        embForPrediction = np.asarray(emb)
        embForPrediction = embForPrediction.reshape(128, 1).T
        pred = model.predict(embForPrediction)
        print("\n\n", pred, "\n\n")
        
        sign.append(emb)
        # embs_array.append(emb)
        
        print("\n\n{}. people" .format(person_number))
        
        #Converting embedding to tensor
        emb = tf.convert_to_tensor([emb])
    
        #Defining input label value due to sex (correct label)
        target_label = np.asarray(input_label_variable)
        target_label = target_label.reshape(4,1).T
        target_label = target_label
            
        #Iterating through adv. examples with different perturbations
        for eps in epsilons_race:
            
            #Setting embModified to initiale value (actual embedding)
            embModified = emb
            embs_array.append(embModified.numpy())
            sign.append(eps)
            
            for numberOfIteration in range(iterationNumber):
                #Creating perturbation
                perturbations = create_adversarial_pattern(embModified, target_label)
                per_array.append((perturbations).numpy())
                #Creating adversarial example
                embModified = embModified + (eps * perturbations)
                embs_array.append(embModified.numpy())
                
            
            #Counting embedding number
            embedding_number = embedding_number + 1
                
            #Predicting sex for advex
            predictedRace = int(model.predict_classes([embModified]))
            prediction_advex_array.append(predictedRace)
                
            if (predictedRace == 0):
                predictedRace = "white"
            if (predictedRace == 1):
                predictedRace = "black"
            if predictedRace == 2:
                predictedRace = "asian"
            if predictedRace == 3:
                predictedRace = "indian"
                
                
            print(eps, ": ", "predicted race for emb:", predictedRace, "ground truth:", dataset[key]["race"])
                
            #Store face embedding
            face_embeddings.append(embModified[0])
                
            #Identification for original emb and advex
            classifier_emb = int(classifiers["{}" .format(key)].predict(emb))
            classifier_emb_array.append(classifier_emb)
            classifier_advex_emb = int(classifiers["{}" .format(key)].predict(embModified))
            classifier_advex_emb_array.append(classifier_advex_emb)
        
            print("Classification for emb: ", classifier_emb, " Classification for advex emb: ", classifier_advex_emb)
            
            if(predictedRace == "white"):
                if(target_label_string_new == "white"):
                    array_white_suc.append(1)
                else:
                    array_white_suc.append(0)
            if(predictedRace == "asian"):
                if(target_label_string_new == "asian"):
                    array_asian_suc.append(1)
                else:
                    array_asian_suc.append(0)
            if(predictedRace == "black"):
                if(target_label_string_new == "black"):
                    array_black_suc.append(1)
                else:
                    array_black_suc.append(0)
            if(predictedRace == "indian"):
                if(target_label_string_new == "indian"):
                    array_indian_suc.append(1)
                else:
                    array_indian_suc.append(0)
            
            if(target_label_string_new == predictedRace):
                predSuccessful[eps].append(1)
                            
                if(classifier_advex_emb == 1):
                    classSuccessful[eps].append(1)
                else:
                    classSuccessful[eps].append(0)
            else:
                predSuccessful[eps].append(0)

    
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


print(np.mean(class_error_final))

cntForEps = 0

wb = Workbook()
results_excel = wb.add_sheet('Results')
results_excel.write(0, 0, 'Epsilon')
results_excel.write(0, 1, 'Successful classification')
results_excel.write(0, 2, 'Successful identification')
results_excel.write(0, cntForEps+8, 'Iteration number')
results_excel.write(0, cntForEps+9, iterationNumber)
results_excel.write(0, cntForEps+12, 'Epsilon values')


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

with os.scandir('D:/Szakdolgozat/Excel/IFGSM/') as entries:
    cnt = 1
    for entry in entries:
        cnt = cnt + 1

nameSave = "D:\Szakdolgozat\Excel\IFGSM\Results_{}". format(cnt) + ".xls"
wb.save(nameSave)

# justTestw = []
# justTestb = []
# justTesta = []
# justTesti = []

# raceSplit = []

# for key in dataset.keys():
#     for emb in dataset[key]["embeddings"]:
#             emb = np.asarray(emb)
#             emb = emb.reshape(128, 1).T
#             res = model.predict_classes(emb)
            
#             raceSplit.append(dataset[key]["race"])
            
#             if(res == 0):
#                 res = "white"
#                 if(dataset[key]["race"] == "white"):
#                     justTestw.append(1)
#                 else:
#                     justTestw.append(0)
#             if(res == 1):
#                 res = "black"
#                 if(dataset[key]["race"] == "black"):
#                     justTestb.append(1)
#                 else:
#                     justTestb.append(0)
#             if(res == 2):
#                 res = "asian"
#                 if(dataset[key]["race"] == "asian"):
#                     justTesta.append(1)
#                 else:
#                     justTesta.append(0)
#             if(res == 3):
#                 res = "indian"
#                 if(dataset[key]["race"] == "indian"):
#                     justTesti.append(1)
#                 else:
#                     justTesti.append(0)
                
            
            
#             print("truth: ", dataset[key]["race"], " prediction: ", res)
            
# justTestw = np.asarray(justTestw)
# print(justTestw[justTestw == 1].size / justTestw.size)

# justTestb = np.asarray(justTestb)
# print(justTestb[justTestb == 1].size / justTestb.size)

# justTesti = np.asarray(justTesti)
# print(justTesti[justTesti == 1].size / justTesti.size)

# justTesta = np.asarray(justTesta)
# print(justTesta[justTesta == 1].size / justTesta.size)

