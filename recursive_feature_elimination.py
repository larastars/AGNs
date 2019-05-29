#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:03:41 2019

@author: larakamal
"""

import csv 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')


def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #logistic regression 
    model = LogisticRegression()
    rfe = RFE(model,k)
    
    fit = rfe.fit(x_train, y_train)
    
    #test
    y_score = rfe.predict(x_test)
    
    #print(y_test)
   # print(y_score)
    
    print('scores:')
    print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
    print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
    print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
    print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
    print(' ')
   

#features to use
k = 10
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
#Feature Extraction 
#model = LogisticRegression()
#rfe = RFE(model,k)

#fit = rfe.fit(training, target)


#print result 
"""
pair = []
for i in range(len(names)):
    pair.append([names[i], fit.ranking_[i], fit.support_[i]])

pair = sorted(pair, key=lambda x: x[1])


############################# PRINTING DATA #############################
print("Num Features: %d"% fit.n_features_) 
for i in range(len(pair)):
    if pair[i][2] == True:
        print(pair[i][0])

"""
cross_validation(5,training,target)

