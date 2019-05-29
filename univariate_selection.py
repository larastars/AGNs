#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:03:41 2019

@author: larakamal
"""

import csv 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
file_reader = open('RatiosGrid_test3.csv', "r", encoding= "ascii")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    target.append(row[:1])
    training.append(row[3:])

#remove the labelling row 
[names] = training[:1]
training = training[1:]
target = target[1:]         


############################# PREPROCESS DATA #############################
#data is stored as string rather than float so we have to conver them
#there are some missing data so we handle that by placing 
#999 in that spot 

for i in range(len(training)):
    for j in range(len(training[1])):
        try:
            training[i][j] = float(training[i][j]) 
        except:
            training[i][j] = float(999)
   
 
for i in range(len(target)):
    target[i] = target[i][0]
    

############################# TRAIN THE DATA #############################
#We will select the features using chi square
test = SelectKBest(chi2,k='all')

#Fit the function for ranking the features by score
fit = test.fit(training, target)

#Summarize scores numpy.set_printoptions(precision=3) print(fit.scores_)
#Apply the transformation on to dataset
features = fit.transform(training)

#Summarize selected features print(features[0:5,:])
np.set_printoptions(precision=3) 
scores_list = fit.scores_


#create a list of pairs [feature, score]
pair = []
for i in range(len(names)):
    pair.append([names[i], scores_list[i]])

pair = sorted(pair, key=lambda x: x[1], reverse=True)


############################# PRINTING DATA #############################
print('features are ranked from the most important to determine the mass of the blackhole to the least important')
for i in range(len(pair)):
    print(pair[i][0], '  %.2f' %pair[i][1])




