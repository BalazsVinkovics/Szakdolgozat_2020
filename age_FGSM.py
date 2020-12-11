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
dataset = pickle.load(open("/content/drive/MyDrive/Szakdolgozat/Data/dataset", "rb" ))

#Creating array for storage of internal values
predictions_asd = []
sign = []
grad = []
classifiers = {}
iterationNumber = 1
loss_stored = []

transability = []

#Dataset keys for test
forTest = [1325, 84]

#Importing classifiers by ids
for index in dataset.keys():
    path = "/content/drive/MyDrive/Szakdolgozat/Classifiers/classifier_id_{}". format(index)
    name = "classifier{}". format(index)
    name = pickle.load(open(path, 'rb'))
    classifiers['{}' .format(index)] = name

# Load the tensorflow model
model = keras.models.load_model('/content/drive/MyDrive/Szakdolgozat/Neural_Network_Age/DNN_race')
model.trainable = False

with open("/content/drive/MyDrive/Szakdolgozat/SVM_AGE/classifier", 'rb') as pickle_file:
    model_SVM = pickle.load(pickle_file)

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
    signed_grad = tf.sign(gradient)
    sign.append(signed_grad.numpy())
    return signed_grad

#Choosing datakey (= person)
# ID = 1

#Creating OneHotEncoding target labels
prediction_0_20 = [1.0, 0.0, 0.0, 0.0]
prediction_20_40 = [0.0, 1.0, 0.0, 0.0]
prediction_40_60 = [0.0, 0.0, 1.0, 0.0]
prediction_60_80 = [0.0, 0.0, 0.0, 1.0]

#Setting epsilon values
epsilons_race = [0.005, 0.0052, 0.0054, 0.0056, 0.0058, 0.006, 0.0062, 0.0064, 0.0066, 0.0068, 0.007]

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

target_0_20 = "0_20"
target_20_40 = "20_40"
target_40_60 = "40_60"
target_60_80 = "60_80"

flag = 1

svmResults = {}



for key in dataset.keys():
    svmResults['{}'. format(key)] = {}

#Iterating through persons by keys, for test we use forTest array wit dedicated keys (only 1325 for now)
for key in dataset.keys():
    
    svmResultsByKey = {}
    for eps in epsilons_race:
        svmResultsByKey['{}'. format(eps)] = []
    
    age_min = int(dataset[key]["age"].split(",")[0])
    age_max = int(dataset[key]["age"].split(",")[1])
    
    sign.append(key)
    
    #Creating arrays for data storage, tracking embedding number
    classifier_emb_array = []
    classifier_advex_emb_array = []
    
    #Tracking embedding number
    embedding_number = 0
    
    #Creating target label with initial value
    input_label_variable = [0.0, 0.0, 0.0, 0.0]
    
    #Setting target label due to race
    if 0 <= age_min:
        if age_max <= 20:
            input_label_variable = prediction_0_20
            target_label_string_new = target_0_20
            dataset[key]["age"] = "0_20"
    if 20 <= age_min:
        if age_max <= 40:
            input_label_variable = prediction_20_40
            target_label_string_new = target_20_40
            dataset[key]["age"] = "20_40"
    if 40 <= age_min:
        if age_max <= 60:
            input_label_variable = prediction_40_60
            target_label_string_new = target_40_60
            dataset[key]["age"] = "40_60"
    if 60 <= age_min:
        if age_max <= 80:
            input_label_variable = prediction_60_80
            target_label_string_new = target_60_80
            dataset[key]["age"] = "60_80"
    
    #Iterating through the chosen person's embeddings
    for emb in dataset[key]["embeddings"]:
        
        embForPrediction = np.asarray(emb)
        embForPrediction = embForPrediction.reshape(128, 1).T
        pred = model.predict(embForPrediction)
        grad.append(pred)
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
            svmPredict = model_SVM.predict(embModified)
            svmResultsByKey["{}". format(eps)].append(svmPredict)
                
            if (predictedRace == 0):
                predictedRace = "0_20"
            if (predictedRace == 1):
                predictedRace = "20_40"
            if predictedRace == 2:
                predictedRace = "40_60"
            if predictedRace == 3:
                predictedRace = "60_80"

            if (svmPredict == 1):
                svmPredict = "0_20"
            if (svmPredict == 2):
                svmPredict = "20_40"
            if svmPredict == 3:
                svmPredict = "40_60"
            if svmPredict == 4:
                svmPredict = "60_80"
                
                
            print(eps, ": ", "predicted race for emb:", predictedRace, "ground truth:", dataset[key]["age"], " SVM: ", svmPredict, " target label ", target_label_string_new)
                
            #Store face embedding
            face_embeddings.append(embModified[0])
                
            #Identification for original emb and advex
            classifier_emb = int(classifiers["{}" .format(key)].predict(emb))
            classifier_emb_array.append(classifier_emb)
            classifier_advex_emb = int(classifiers["{}" .format(key)].predict(embModified))
            classifier_advex_emb_array.append(classifier_advex_emb)

            print("Classification for emb: ", classifier_emb, " Classification for advex emb: ", classifier_advex_emb)
            
            if(predictedRace != target_label_string_new):
                if svmPredict != target_label_string_new:
                  transability.append(1)
                else:
                  transability.append(0)
            
            if(predictedRace == "0_20"):
                if(target_label_string_new != "0_20"):
                    array_white_suc.append(1)
                else:
                    array_white_suc.append(0)
            if(predictedRace == "20_40"):
                if(target_label_string_new != "20_40"):
                    array_asian_suc.append(1)
                else:
                    array_asian_suc.append(0)
            if(predictedRace == "40_60"):
                if(target_label_string_new != "40_60"):
                    array_black_suc.append(1)
                else:
                    array_black_suc.append(0)
            if(predictedRace == "60_80"):
                if(target_label_string_new != "60_80"):
                    array_indian_suc.append(1)
                else:
                    array_indian_suc.append(0)
            
            if(target_label_string_new != predictedRace):
                predSuccessful[eps].append(1)
                            
                if(classifier_advex_emb == 1):
                    classSuccessful[eps].append(1)
                else:
                    classSuccessful[eps].append(0)
            else:
                predSuccessful[eps].append(0)

    
    #Calculating error of identification due advex  
    svmResults["{}". format(key)] = svmResultsByKey
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
results_excel.write(0, 5, 'Ground truth')
results_excel.write(0, 6, 'Target label')
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
    if (array_ActClass.size != 0):
        resultOfClassification = array_ActClass[array_ActClass == 1].size / array_ActClass.size
    else:
        resultOfClassification = 0
    print(resultOfClassification)
    results_excel.write(lineCounter, 0, eps)
    results_excel.write(lineCounter, 1, resultOfClassification)
    lineCounter = lineCounter  + 1
    
lineCounter = 1
    
for eps in epsilons_race:
    array_ActClass = predSuccessful[eps]
    array_ActClass = np.asarray(array_ActClass)
    print("successfull prediction for eps: ", eps)
    if(array_ActClass.size != 0):
        resultOfIdentification = array_ActClass[array_ActClass == 1].size / array_ActClass.size
    else:
        resultOfIdentification = 0
    print(resultOfIdentification)
    results_excel.write(lineCounter, 2, resultOfIdentification)
    lineCounter = lineCounter + 1

finalSVM = {}
for eps in epsilons_race:
    finalSVM['{}'. format(eps)] = []

for key in forTest:
    array = svmResults["{}". format(key)]
    for embs in array:
        for eps in epsilons_race:
            for emb in embs:
                if(dataset[key]["age"] == "0_20"):
                    if(emb[0] == 2):
                        finalSVM["{}". format(eps)].append(1)
                    else:
                        finalSVM["{}". format(eps)].append(0)
                if(dataset[key]["age"] == "20_40"):
                    if(emb[0] == 1):
                        finalSVM["{}". format(eps)].append(1)
                    else:
                        finalSVM["{}". format(eps)].append(0)
                if(dataset[key]["age"] == "40_60"):
                    if(emb[0] == 4):
                        finalSVM["{}". format(eps)].append(1)
                    else:
                        finalSVM["{}". format(eps)].append(0)
                if(dataset[key]["age"] == "60_80"):
                    if(emb[0] == 3):
                        finalSVM["{}". format(eps)].append(1)
                    else:
                        finalSVM["{}". format(eps)].append(0)


with os.scandir('/content/drive/MyDrive/Szakdolgozat/Results/Final/FGSM/Age') as entries:
    cnt = 1
    for entry in entries:
        cnt = cnt + 1
    
nameSave = "/content/drive/MyDrive/Szakdolgozat/Results/Final/FGSM/Age/Results_{}". format(cnt) + ".xls"
wb.save(nameSave)

justTestw = []
justTestb = []
justTesta = []
justTesti = []

raceSplit = []

# for key in dataset.keys():
#     for emb in dataset[key]["embeddings"]:
#             emb = np.asarray(emb)
#             emb = emb.reshape(128, 1).T
#             res = model.predict_classes(emb)
            
#             raceSplit.append(dataset[key]["age"])
            
#             if(res == 0):
#                 res = "0_20"
#                 if(dataset[key]["age"] == "0_20"):
#                     justTestw.append(1)
#                 else:
#                     justTestw.append(0)
#             if(res == 1):
#                 res = "20_40"
#                 if(dataset[key]["age"] == "20_40"):
#                     justTestb.append(1)
#                 else:
#                     justTestb.append(0)
#             if(res == 2):
#                 res = "40_60"
#                 if(dataset[key]["age"] == "40_60"):
#                     justTesta.append(1)
#                 else:
#                     justTesta.append(0)
#             if(res == 3):
#                 res = "60_80"
#                 if(dataset[key]["age"] == "60_80"):
#                     justTesti.append(1)
#                 else:
#                     justTesti.append(0)
                
            
            
#             print("truth: ", dataset[key]["age"], " prediction: ", res)
            
# justTestw = np.asarray(justTestw)
# print(justTestw[justTestw == 1].size / justTestw.size)

# justTestb = np.asarray(justTestb)
# print(justTestb[justTestb == 1].size / justTestb.size)

# justTesti = np.asarray(justTesti)
# print(justTesti[justTesti == 1].size / justTesti.size)

# justTesta = np.asarray(justTesta)
# print(justTesta[justTesta == 1].size / justTesta.size)

