# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:47:05 2020

@author: Balázs Vinkovics
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

#Importing dataset
dataset = pickle.load(open("D:/Szakdolgozat/10_11/Data/dataset", "rb" ))

#Creating array for storage of internal values
predictions_asd = []
sign = []
grad = []
classifiers = {}

#Importing demographic data classifiers
race_model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks2/DNN_race')
sex_model = keras.models.load_model('D:/Szakdolgozat/AdvEx_v2/DNN_Classifier/sex.model2')
age_model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks3/DNN_race')

#Importing classifiers by ids
for index in dataset.keys():
    path = "D:/Szakdolgozat/10_11/Data/Classifiers/classifier_id_{}". format(index)
    name = "classifier{}". format(index)
    name = pickle.load(open(path, 'rb'))
    classifiers['{}' .format(index)] = name

# Load the tensorflow model
model = keras.models.load_model('D:/Szakdolgozat/Neural_Networks2/DNN_race')
model.trainable = False

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

prediction_female = [1.0, 0.0]
prediction_male = [0.0, 1.0]

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
    
    #Iterating through the chosen person's embeddings
    for emb in dataset[key]["embeddings"]:
        print("\n\n{}. people" .format(person_number))
        #Converting embedding to tensor
        emb = tf.convert_to_tensor([emb])
    
        #Defining input label value due to sex (correct label)
        per_array_sex = []
        per_array_race = []
        per_array_age = []
        
        target_label_race = np.asarray(input_label_variable_race)
        target_label_race = target_label_race.reshape(4, 1).T
        
        target_label_sex = np.asarray(input_label_variable_sex)
        target_label_sex = target_label_sex.reshape(2,1).T
        
        target_label_age = np.asarray(input_label_variable_age)
        target_label_age = target_label_age.reshape(4,1).T
        
        #Counting embedding number
        embedding_number = embedding_number + 1
        
        embForEps = []
        cntEps = 0
        
        #Iterating through adv. examples with different perturbations
        for eps in epsilons_sex:
            
            #Setting embModified to initiale value (actual embedding)
            embModified_sex = emb
            cnt_sex = 0
            embModified_race = emb
            cnt_race = 0
            #Setting embModified to initiale value (actual embedding)
            embModified_age = emb
            cnt_age = 0
            
            perturbation_sex = []
            perturbation_age = []
            perturbation_race = []
            
            for numberOfIteration in range(iterationNumber_sex):
                #Creating perturbation
                perturbation_sex = create_adversarial_pattern_sex(embModified_sex, target_label_sex)
                #Creating adversarial example
                embModified_sex = embModified_sex + (eps * perturbation_sex)
                cnt_sex = cnt_sex + 1
                if cnt_sex == iterationNumber_sex:
                    perturbation_sex = embModified_sex - emb
                    per_array_sex.append((embModified_sex).numpy())
                    targets_sex.append(target_label_string_sex)
                    print("Sex advex generated")
                    
            for numberOfIteration in range(iterationNumber_race):
                #Creating perturbation
                perturbation_race = create_adversarial_pattern_race(embModified_race, target_label_race)
                #Creating adversarial example
                embModified_race = embModified_race + (epsilons_race[cntEps] * perturbation_race)
                cnt_race = cnt_race + 1
                if cnt_race == iterationNumber_race:
                    perturbation_race = embModified_race - emb
                    per_array_race.append((embModified_race).numpy())
                    targets_race.append(target_label_string_race)
                    print("Race advex generated")
                    
            for numberOfIteration in range(iterationNumber_age):
                #Creating perturbation
                perturbation_age = create_adversarial_pattern_age(embModified_age, target_label_age)
                #Creating adversarial example
                embModified_age = embModified_age + (epsilons_age[cntEps] * perturbation_age)
                cnt_age = cnt_age + 1
                if cnt_age == iterationNumber_age:
                    perturbation_age = embModified_age - emb
                    per_array_age.append((embModified_age).numpy())
                    targets_age.append(target_label_string_age)
                    print("Age advex generated")
        
            averageEmbedding = emb + perturbation_sex + perturbation_age + perturbation_race
            cntEps = cntEps + 1
            embForEps.append(averageEmbedding.numpy())
        embForEmbs.append(embForEps)
    embForKeys.append(embForEmbs)

    
# cntEps = len(epsilons_age)
# embs_final = []
# for embs in embForKeys:
#     embs_eps_temp = []
#     for embs2 in embs:
#         embs_eps = []
#         for cntFeature in range(128):
#             embs_temp = []
#             for cnteps in range(cntEps):
#                 embs_temp.append(embs2[cnteps][0][cntFeature])
#             embs_eps.append(embs_temp)
#         embs_eps_temp.append(embs_eps)
#     embs_final.append(embs_eps_temp)
        

resultsSex = {}
resultsAge = {}
resultsRace = {}

for epsilons in epsilons_sex:
    resultsSex['{}'. format(epsilons)] = []
    
for epsilons in epsilons_race:
    resultsRace['{}'. format(epsilons)] = []
    
for epsilons in epsilons_age:
    resultsAge['{}'. format(epsilons)] = []

for embs in embForKeys:
    cnt = 0    
    for embs2 in embs:
        for cntEps in range(len(embs2)):
            actEmb= np.asarray(embs2[cntEps][0])
            actEmb = actEmb.reshape(128, 1).T
            
            sexPredUni = sex_model.predict_classes(actEmb)
            if (sexPredUni == 0):
                sexPredUni = "f"
            if (sexPredUni == 1):
                sexPredUni = "m"    
            
            resultsSex['{}'. format(epsilons_sex[cntEps])].append(sexPredUni)
            
            racePredUni = race_model.predict_classes(actEmb)
            if (racePredUni == 0):
                racePredUni = "white"
            if (racePredUni == 1):
                racePredUni = "black"
            if (racePredUni == 2):
                racePredUni = "asian"
            if (racePredUni == 3):
                racePredUni = "indian"
            
            resultsRace['{}'. format(epsilons_race[cntEps])].append(racePredUni)
        
            predict_age = age_model.predict(embs2[cntEps])
        
            agePredUni = age_model.predict_classes(actEmb)
            if (agePredUni == 0):
                agePredUni = "0_20"
            if (agePredUni == 1):
                agePredUni = "20_40"
            if agePredUni == 2:
                agePredUni = "40_60"
            if agePredUni == 3:
                agePredUni = "60_80"
            
            resultsAge['{}'. format(epsilons_age[cntEps])].append(agePredUni)
        
        cnt = cnt + 1
 
finalResultsAge = {}
finalResultsRace = {}
finalResultsSex = {}       
 
for epsilons in epsilons_sex:
    finalResultsSex['{}'. format(epsilons)] = []
    
for epsilons in epsilons_race:
    finalResultsRace['{}'. format(epsilons)] = []
    
for epsilons in epsilons_age:
    finalResultsAge['{}'. format(epsilons)] = []
        
for eps in epsilons_age:
    cntAll_age = 0
    for key in keys_stored:
        for cntNumber in range (len(dataset[key]["embeddings"])):
            print(resultsAge["{}". format(eps)][cntAll_age], "==", dataset[key]["age"], " keys: ", key)
            if resultsAge["{}". format(eps)][cntAll_age] == dataset[key]["age"]:
                finalResultsAge["{}". format(eps)].append(1)
                cntAll_age = cntAll_age + 1
                print(1)
            else:
                finalResultsAge["{}". format(eps)].append(0)
                cntAll_age = cntAll_age + 1
                print(0)
                
for eps in epsilons_sex:
    cntAll_sex = 0
    for key in keys_stored:
        for cntNumber in range (len(dataset[key]["embeddings"])):
            print(resultsSex["{}". format(eps)][cntAll_sex], "==", dataset[key]["sex"], " keys: ", key)
            if resultsSex["{}". format(eps)][cntAll_sex] == dataset[key]["sex"]:
                finalResultsSex["{}". format(eps)].append(1)
                cntAll_sex = cntAll_sex + 1
                print(1)
            else:
                finalResultsSex["{}". format(eps)].append(0)
                cntAll_sex = cntAll_sex + 1
                print(0)
            
for eps in epsilons_race:
    cntAll_race = 0 
    for key in keys_stored:
        for cntNumber in range (len(dataset[key]["embeddings"])):
            # print(resultsRace["{}". format(eps)][cntAll_race], "==", dataset[key]["race"], " keys: ", key)
            if resultsRace["{}". format(eps)][cntAll_race] == dataset[key]["race"]:
                finalResultsRace["{}". format(eps)].append(1)
                cntAll_race = cntAll_race + 1
                # print(1)
            else:
                finalResultsRace["{}". format(eps)].append(0)
                cntAll_race = cntAll_race + 1
                # print(0)
                
for eps in epsilons_sex:
    resultsArraySex = finalResultsSex["{}". format(eps)]
    resultsArraySex = np.asarray(resultsArraySex)
    percent = resultsArraySex[resultsArraySex == 1].size / resultsArraySex.size
    print(eps," : ", percent)
    
for eps in epsilons_race:
    resultsArrayRace = finalResultsRace["{}". format(eps)]
    resultsArrayRace = np.asarray(resultsArrayRace)
    percent = resultsArrayRace[resultsArrayRace == 1].size / resultsArrayRace.size
    print(eps," : ", percent)
 
for eps in epsilons_age:
    resultsArrayAge = finalResultsAge["{}". format(eps)]
    resultsArrayAge = np.asarray(resultsArrayAge)
    percent = resultsArrayAge[resultsArrayAge == 1].size / resultsArrayAge.size
    print(eps," : ", percent)
       
# newEmb = []
# for numberOfEmb in range(len(per_array_age)):
#     universalEmbedding = []
#     for numberOfElements in range(per_array_age[0].size):
#         element = (per_array_age[numberOfEmb][0][numberOfElements] + per_array_sex[numberOfEmb][0][numberOfElements] + per_array_race[numberOfEmb][0][numberOfElements]) / 3.0
#         universalEmbedding.append(element)
#     universalEmbeddings.append(universalEmbedding)

# cnt_uniEmb = 0  
# # keys_stored = np.asarray(keys_stored)      
# for uniEmb in universalEmbeddings:
    
#     uniEmb = np.asarray(uniEmb)
#     uniEmb = uniEmb.reshape(128, 1).T
    
#     sexPredUni = sex_model.predict_classes(uniEmb)
#     if (sexPredUni == 0):
#         sexPredUni = "f"
#     if (sexPredUni == 1):
#         sexPredUni = "m"    
    
#     agePredUni = age_model.predict_classes(uniEmb)
#     if (agePredUni == 0):
#         agePredUni = "0_20"
#     if (agePredUni == 1):
#         agePredUni = "20_40"
#     if agePredUni == 2:
#         agePredUni = "40_60"
#     if agePredUni == 3:
#         agePredUni = "60_80"
    
#     racePredUni = race_model.predict_classes(uniEmb)
#     if (racePredUni == 0):
#         racePredUni = "white"
#     if (racePredUni == 1):
#         racePredUni = "black"
#     if (racePredUni == 2):
#         racePredUni = "asian"
#     if (racePredUni == 3):
#         racePredUni = "indian"
    
#     if (targets_race[cnt_uniEmb] == racePredUni):
#         predSucRace.append(1)
#     else:
#         predSucRace.append(0)
        
#     if (targets_sex[cnt_uniEmb] == sexPredUni):
#         predSucSex.append(1)
#     else:
#         predSucSex.append(0)
        
#     if (targets_age[cnt_uniEmb] == agePredUni):
#         predSucAge.append(1)
#     else:
#         predSucAge.append(0)
        
#     #Identification for original emb and advex
#     classifier_advex_emb = int(classifiers["{}" .format(keys_stored[cnt_uniEmb])].predict(uniEmb))
#     classifier_advex_emb_array.append(classifier_advex_emb)
#     classifier_advex_emb_array = np.asarray(classifier_advex_emb_array)
    
#     cnt_uniEmb = cnt_uniEmb + 1
        