#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:11:52 2019

@author: larakamal
"""

import csv 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import math 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import rcParams
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import pandas as pd
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')
import time


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

#convert to log function 
def logFunction(list):
    list2 = toFloat(list)
    result = []
    for i in list2:
        try:
            ans = round(math.log(i,10),1)
        except:
            ans = 0.0
        result.append(ans)
        #time.sleep(0.1)
    return result


#convert a list to float 
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


def removeData(training, names):
    #if the average = first value 
    training2 = matrixTranspose(training)
    names2 = matrixTranspose(names)
    
    training3 = []
    names3 = []
    
    for i in range(len(training2)):
        avg = sum(training2[i])/ float(len(training2[i]))
        if (avg != training2[i][0]):
            training3.append(training2[i])
            names3.append(names2[i])
          
    training4 = matrixTranspose(training3)
    names4 = matrixTranspose(names3)
    [names5] = names4
   

    return training4, names5
                           

def breakInBins(training, target, targetu):
    training1 = []
    target1 = []
    targetu1 = []
    training2= []
    target2 = []
    targetu2 = []
    training3 = []
    target3 = []
    targetu3 = []
    
    floatTarget = toFloat(target)
    for i in range(len(target)):
        if (floatTarget[i] <= 4 ):
            training1.append(training[i])
            target1.append(target[i])
            targetu1.append(targetu[i])
        elif (floatTarget[i] > 4 and floatTarget[i] <= 6):
            training2.append(training[i])
            target2.append(target[i])
            targetu2.append(targetu[i])
        else:
            training3.append(training[i])
            target3.append(target[i])
            targetu3.append(targetu[i])
    return training1, target1, targetu1, training2, target2, targetu2, training3, target3, targetu3

def plotSingle(x,y,xname,yname):
     fig, ax = plt.subplots()
    
     ax.plot(x, y,'o')
        
     ax.set(ylabel= yname,xlabel=xname)
     plt.title(xname + ' vs. '+ yname)
        
     ax.grid()
     plt.show()
     rcParams.update({'figure.autolayout': True})
     fileName = 'HighMassa/' + (yname+ ' ' +xname).replace('/','').replace('1.01','1')
    
     #print(fileName)
     fig.savefig(fileName)   # save the figure to file
     
     plt.close(fig)    # close the figure
    

def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    #convert training to float
    trainingFloat = []
    for i in training:
        trainingFloat.append(toFloat(i))
        
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    lasso = Lasso(alpha=0.000001, max_iter =7000,random_state=None,tol=0.06)
    lasso.fit(x_train,y_train)
    
    #test
    y_score = lasso.predict(x_test)
    
    mse = mean_squared_error(toFloat(y_test), toFloat(y_score))
    print(mse)
    
def applyRegression(training, target, names):
  
    elementList = []
   
    y = np.array(toFloat(target))
    x = []
    #convert training to float
    for i in training:
        x.append(toFloat(i))
    
    lasso = Lasso(alpha=0.000001, max_iter =7000,random_state=None,tol=0.06)
    lasso.fit(x,y)
    
    weight = np.array(lasso.coef_)
   
    for i in range(len(names)):
        elementList.append([names[i],abs(weight[i]),0])
    
    #print(np.sum(lasso.coef_ !=0))
    #print(np.array(lasso.coef_))
    
    #[name,weight,0]
    return elementList

    
def getResult(MassBin, uListBin):
    
    final = []
    listMass = []
    listU = [] 

    for i in MassBin:
        listMass.append([i[0],i[1]+i[2]])    
    listMass.sort(key=lambda x: x[1], reverse =True)

    for i in uListBin:
        listU.append([i[0],i[1]+i[2]])     
    listU.sort(key=lambda x: x[1], reverse =True)

    k = 25
    listMass = listMass[:k]
    listU = listU[:k]
    
    for i in listMass:
       n = len(listMass) -1
       for j in listU:
           if (i[0] == j[0]):
               break
           elif (n == 0):
               final.append(i)
           n -=1 
    
    #print(listMass)
    #print("")
    #print(listU)
    #print("")
    #print(final)
    #print("")
    #print("")      
    return final 

def printToFile(list1, list2, list3):
    maxLen = max(len(list1), len(list2), len(list3))
    list1 = pad(list1,0,maxLen)
    list2 = pad(list2,0,maxLen) 
    list3 = pad(list3,0,maxLen)   
    
    with open('output2.csv', mode='w') as file:
        outputwriter = csv.writer(file, delimiter=',')
        outputwriter.writerow(['Low Mass AGNs', 'Medium Mass AGNs','High Mass AGNs'])
        for i in range(maxLen):
            outputwriter.writerow([str(list1[i]), str(list2[i]),str(list3[i])])
    file.close()

def pad(l, content, width):
     l.extend([content] * (width - len(l)))
     return l
 
def contourPlot(x,y,z,xname,yname,zname):
    plt.figure()
    x1, y1 = np.meshgrid(x, y)
    cp = plt.contourf(x, y, z)
    plt.colorbar(cp)
    plt.title(zname)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
    
    
def findSubset(training, target, utarget, names):
   
    #break into bins
    trainingbin1, targetbin1, targetubin1, trainingbin2, targetbin2, targetubin2, trainingbin3, targetbin3, targetubin3 = breakInBins(training, target, utarget)
    
    MassBin1 = applyRegression(trainingbin1, targetbin1, names2)
    
    MassBin2 = applyRegression(trainingbin2, targetbin2, names2)
    
    MassBin3 = applyRegression(trainingbin3, targetbin3, names2)
    
    
    uListBin1 = applyRegression(trainingbin1, targetubin1, names2)
    
    uListBin2 = applyRegression(trainingbin2, targetubin2, names2)
    
    uListBin3 = applyRegression(trainingbin3, targetubin3, names2)
    
    #[name, coef, score]
    
    list1 = getResult(MassBin1, uListBin1)
    list2 = getResult(MassBin2, uListBin2)
    list3 = getResult(MassBin3, uListBin3)
    
    #has [name, final, mass coef, mass mse, u coef, u mse]
    printToFile(list1,list2,list3) 
    return list1,list2,list3

def heatMap(x,y,z,xname,yname,zname):
    z1 = matrixTranspose(z)
    
    #for z2 in z1:
    for j in range(len(z1)):
        i = 0
        z3=[]
        y2=[]
        x2=[]
        while(i <= len(x)-1):
            x2.append(x[i])
            y2.append(y[i])
            z3.append((sum([z1[j][i],z1[j][i+1],z1[j][i+2]])/3))
            i +=3
      
        matrix = list(zip(x2,y2,z3))
        
        #for i in range(len(x2)):
         #   print(x2[i],y2[i],z3[i])
    
        table = pd.DataFrame(matrix, columns=[xname,yname,zname[j]])
        #print(table)
        table = table.pivot(xname,yname,zname[j])
        
        fig, ax = plt.subplots(figsize=(8.5,6))
        ax = sns.heatmap(table)
        ax.invert_yaxis()
        ax.set_title(zname[j])
        fileName = 'Heatmapnew/' + (zname[j]).replace('/','').replace('1.01','1')
        #print(fileName)
        fig.savefig(fileName)   # save the figure to file
         
        plt.close(fig)    # close the figure
        
    
############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
utarget= []
file_reader = open('RatiosGrid_test4.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[:1] != ['']):# and row[3:][0] != ''):
        utarget.append(row[1:2])
        target.append(row[:1])
        training.append(row[3:])
        
    
file_reader.close()       
#remove the labelling row 
[names] = training[:1]
training = training[1:]

target = target[1:] 
utarget = utarget[1:]    

#print(target)
############################# PREPROCESS DATA #############################
#data is stored as string rather than float so we have to conver them
#there are some missing data so we handle that by placing 
#0 in that spot 

for i in range(len(training)):
    for j in range(len(training[1])):
        try:
            training[i][j] = float(training[i][j]) 
        except:
            training[i][j] = 0
  

#transpose a lit
for i in range(len(utarget)):
    utarget[i] = round(float(utarget[i][0]),3)
    
for i in range(len(target)):
    target[i] = target[i][0]

#convert values to float 
for i in range(len(target)):
    target[i] = float(target[i])

#appy log function 
targetlog = logFunction(target)  


#normalize data using zscores
#round the number to 3 decimal place
trans = matrixTranspose(training)
newTraining = []
 
for i in trans:  
    [item] = preprocessing.normalize([i])
    newTraining.append(item)

training2 = matrixTranspose(newTraining)

#remove unuseful data
training3, names2 = removeData(training2, names)

#cross_validation(5,training3,targetlog)
list1,list2,list3  = findSubset(training3, targetlog, utarget, names2)  

heatMap(targetlog,utarget,training2,'Log(Mass)','U',names)
#cross_validation(5,training3,targetlog)

#createGraph(list1,list2,list3,'S9(1)/S3(18)', 'S9(3)/S3(18)')
#createGraph(list1,list2,list3,'Na4(9)/Na3', 'Na4(21)/Na3')

#low mass 
#print('low mass')
#createGraph(list1,list2,list3,'Na6(8)/Na4', 'Si10/Si9')
#createGraph(list1,list2,list3,'Na6(8)/Na4(6)', 'Si11/Si10')
#createGraph(list1,list2,list3,'Na6(14)/Na4(21)', 'Al6(9)/Al5')

#medium mass

#print('medium mass')
#createGraph(list1,list2,list3,'Fe13/Fe6(1.01)', 'Si9/Si6')
#createGraph(list1,list2,list3,'Fe13/Fe6(1.01)', 'Si11/Si6')
#createGraph(list1,list2,list3,'Si9/Si6', 'Al8(5)/Al6(9)')

#high mass 
#print('high mass')
#createGraph(list1,list2,list3,'Si11/Si10', 'Si11/Si6')
#createGraph(list1,list2,list3,'Si11/Si7(2)', 'Si11/Si7(6)')
#createGraph(list1,list2,list3,'Si10/Si6', 'Si9/Si6')




#createDataGraph(list1,list2,list3,MassBin1, MassBin2, MassBin3,'Mg5(5)/Mg4', 'Si10/Si9')

