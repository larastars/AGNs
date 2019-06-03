#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:36:59 2019

@author: larakamal
"""
import csv 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from matplotlib import rcParams
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')

#find the transpose of a matrix 
def matrixTranspose(matrix):
    if not matrix: return []
    return [ [ row[ i ] for row in matrix ] for i in range( len( matrix[ 0 ] ) ) ]

#convert a list to float 
def toFloat(list):
    list2 = []
    for i in list:
        try:
            list2.append(float(i))
        except:
            print('error',i)
            list2.append(0)
    return list2

#convert a list to string 
def toString(list):
    myList = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q']
    list2 = []
    curr = list[0]
    count = 0
    for i in list:
        if (i == curr):
            item = myList[count] + '- ' + str(i)
            list2.append(item)
        else:
            count = count +1
            curr = i
            item = myList[count] + '- ' + str(i)
            list2.append(item)
        
    return list2   

#generate the plots
def plot(training, target, names):
    #find the transpose of training data 
    trainingTrans =  matrixTranspose(training) 
    for i in range(len(trainingTrans)):
        #create an instance of plotting 
        fig, ax = plt.subplots()
        
    
        #convert to float and string 
        trainingInst = toFloat(trainingTrans[i])
        target2 = toString(target)
        
        #find the length of unique data 
        y_len = len(set(target2))
        
        #plt.yticks(range(len(target2)),target2)
        
        #plot 
        ax.plot(trainingInst, target2,'o')
        ax.set(xlabel= names[i],ylabel='Mass of blackhole')
        plt.title(names[i] +' vs. Mass of blackhole')
        
        ax.grid()
        plt.show()
        rcParams.update({'figure.autolayout': True})
        fileName = 'plots/' +str(i+1)
        fig.savefig(fileName)   # save the figure to file
        plt.close(fig)    # close the figure
    
  
def remove_columns(org_order, sorted_order, training,k):
    training2 = []
    index = []
    #top 7
    for i in range(0,k):
       #find item
       curr = sorted_order[i]
       #print(curr)
       index.append(org_order.index(curr))
       
    for i in range(len(training)):
        list = []
        for j in range(len(training[0])):
            if j in index:
                list.append(training[i][j])
        training2.append(list)  

    return training2
    
def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #classifier 
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    #train 
    clf.fit(x_train, y_train)
    #print(clf.feature_importances_)
   
    #test
    y_score = clf.predict(x_test)
    print('scores:')
    print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
    print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
    print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
    print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
    print(' ')
   


############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
file_reader = open('RatiosGrid_test3.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[:1] != '' and row[3:][0] != ''):
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


a#create a list of pairs [feature, score]
pair = []
for i in range(len(names)):
    pair.append([names[i], scores_list[i]])

pair2 = sorted(pair, key=lambda x: x[1], reverse=True)


############################# PRINTING DATA #############################
#print('features are ranked from the most important to determine the mass of the blackhole to the least important')
#for i in range(len(pair2)):
#    print(pair2[i][0], '  %.2f' %pair2[i][1])


#cross_validation(5,training,target)
#training2 = remove_columns(pair, pair2, training,10)
cross_validation(5,training,target)
#plot(training, target, names)
    
    
