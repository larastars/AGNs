#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:16:12 2019

@author: larakamal
"""

import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import math
from sklearn.linear_model import LassoCV
import statistics
import warnings
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')

def toFloat(list):
    list2 = []
    for i in list:
        try:
            list2.append(float(i))
        except:
            #print('error',i)
            [ans] = i
            list2.append(float(ans))
    return list2

#find the transpose of a matrix 
def matrixTranspose(matrix):
    if not matrix: 
        return []
    else:
        try:
            return [ [ row[ i ] for row in matrix ] for i in range( len( matrix[ 0 ] ) ) ]
        except:
            result = []
            for i in matrix:
                result.append([i])
            return result
        
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def removeData(training, names, removeList):
    training2 = matrixTranspose(training)
    
    training3 = []
    names2 = []
    indexList = []
    
    for i in removeList:
        #find the index of 
        index = names.index(i)
        indexList.append(index)
        
    for i in range(len(training2)): 
        if (i not in indexList):
            training3.append(training2[i])
            names2.append(names[i])
    
    training4 = matrixTranspose(training3)
    
    return training4, names2



    
#read file 
training = []
names = []
utarget = []
ztarget = []
ntarget = []
nhtarget = []
agntarget = [] 
rtarget = []
#redshift = []

#open a new file  and store the data in a list 
file_reader = open('combinedFileV4.csv', "r")
read = csv.reader(file_reader)
for row in read:
    if(row[3] != ''):
        #adding the information to a list 
        try: 
            agntarget.append(float(row[0]))
            ztarget.append(float(row[1]))
            ntarget.append(float(row[2]))
            rtarget.append(float(row[3]))
            nhtarget.append(float(row[4]))
            utarget.append(float(row[5]))
            #redshift.append(float(row[6]))
            training.append(row[6:])
        except:
            names.append(row[6:])
        
file_reader.close()   

[names2] = names


#remove optical lines
opticalList = ['H1_4861.36A','O3_5007.00A','H1_6562.85A','N2_6584.00A','O1_6300.00A','S2_6720.00A','NE5_3345.99A','NE5_3426.03A','FE11_7891.87A','S12_7610.59A','FE7_5720.71A','FE7_5720.22A','FE7_6086.97A']
opticalInter = intersection(opticalList,names2)
trainingRem, namesRem = removeData(training,names2,opticalInter)






# find z values 
# Divide every element based on z value 
# find mean and standard dev for those values for both z values
# find out which elements have the highest difference between the z values 

z_1_matrix = []
z_01_matrix = []
#trainingResult3Trans = matrixTranspose(trainingResult3)

for i in range(len(ztarget)):
    if (ztarget[i] == 1):
        z_1_matrix.append(toFloat(trainingRem[i]))
    else:
        z_01_matrix.append(toFloat(trainingRem[i]))

z_1_trans = matrixTranspose(z_1_matrix)     
z_01_trans = matrixTranspose(z_01_matrix)  

#find the mean and stdev of z1
z_1_mean =[]
z_1_stdv =[]
for i in z_1_trans:
    curr = toFloat(i)
    avg = statistics.mean(curr)
    z_1_mean.append(avg)
    stdv = statistics.stdev(curr)
    z_1_stdv.append(stdv)
    
#find the mean and stdev of z01  
z_01_mean =[]
z_01_stdv =[]
for i in z_01_trans:
    curr = toFloat(i)
    avg = statistics.mean(curr)
    z_01_mean.append(avg)
    stdv = statistics.stdev(curr)
    z_01_stdv.append(stdv)

compare = []
#find out which is larger 
#mean z1 or mean z01
#larger = num - stdv
#smaller = num + stdv
#subtract larger - smaller 
for i in range(len(z_1_mean)):
    larger = 0
    smaller = 0
    if (z_1_mean[i] > z_01_mean[i]):
        larger = z_1_mean[i] - z_1_stdv[i]
        smaller = z_01_mean[i] + z_01_stdv[i]
    else:
        larger = z_01_mean[i] - z_01_stdv[i]
        smaller = z_1_mean[i] + z_1_stdv[i]
    
    compare.append(larger-smaller)
    

compare, namesRem = zip(*sorted(zip(compare, namesRem), reverse=False))

#compare = list(compare)
namesRem = list(namesRem)

print('')  
print(compare)    
print(namesRem)

    

       



