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
import warnings
import math 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from matplotlib import rcParams
from scipy import stats
from sklearn import preprocessing

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
    [names4] = matrixTranspose(names3)

    return training4, names4
                           

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
    

def cross_validation(k,training,target,names):
    fold = 100/k
    fold = fold/100
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #larger alpha = less coef 
    lasso = Lasso(alpha=0.3, max_iter =46000)
    lasso.fit(x_train,y_train)
    
    #test
    y_score = lasso.predict(x_test)
    
    #find the weights 
    weight = np.array(lasso.coef_)
    elementList = []
    for i in range(len(names)):
        elementList.append([names[i],weight[i]])
    
    #sort the list based on weight 
    elementList.sort(key=lambda x: x[1], reverse =True)
    
  
    #nonzero= np.sum(lasso.coef_ != 0.0)
    elementList2 = []
    for i in elementList:
        if (i[1] > 0.0 or i[1] < 0.0 ):
            elementList2.append(i)
  
    nonzero = len(elementList2)  
    #find mse
    mse = mean_squared_error(toFloat(y_test), toFloat(y_score))
  
    print(elementList2)
    print('')
    print('# of coeff = ', nonzero)
    print('mse = ', mse)
    print('')
    
def applyRegression(training, target, names):
  
    elementList = []
   
    y = np.array(toFloat(target))
    x = []
    #convert training to float
    for i in training:
        x.append(toFloat(i))
    
    lasso = Lasso(alpha=0.000001, max_iter =46000)
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
    list1 = pad(list1,'',maxLen)
    list2 = pad(list2,'',maxLen) 
    list3 = pad(list3,'',maxLen)   
    
    with open('output3.csv', mode='w') as file:
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

def boxcox(x):
    xt =[]
    try:
        xt, maxlog, interval = stats.boxcox(x, alpha=0.05)
    except:
        for i in range(len(x)):
            if (x[i] == 0):
                x[i]=0.0000000001
        xt, maxlog, interval = stats.boxcox(x, alpha=0.05)
    return xt     
    
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

    #heatMap(targetbin1,targetubin1,trainingbin1,'Log(Mass)','U',names, 'Heatmaplow')
    #heatMap(targetbin2,targetubin2,trainingbin2,'Log(Mass)','U',names, 'Heatmapmid')
    #heatMap(targetbin3,targetubin3,trainingbin3,'Log(Mass)','U',names, 'Heatmaphigh')


    #has [name, final, mass coef, mass mse, u coef, u mse]
    printToFile(list1,list2,list3) 
    return list1,list2,list3

def heatMap(x,y,z,xname,yname,zname,filename):
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
        #8.5,6
        fig, ax = plt.subplots(figsize=(13,3.5))
        ax = sns.heatmap(table)
        ax.invert_yaxis()
        ax.set_title(zname[j])
        fileName = filename + '/' + (zname[j]).replace('/','').replace('1.01','1')
        #print(fileName)
        fig.savefig(fileName)   # save the figure to file
         
        plt.close(fig)    # close the figure

############################# READ TRAINING DATA #############################
"""      
#read file 
training = []
names = []
utarget = []
ztarget = []
ntarget = []
nhtarget = []
agntarget = [] 
rtarget = []

file_reader = open('colorgridoutput6.csv', "r")
read = csv.reader(file_reader)
for row in read:
    if(row[3] != ''):
       
        try: 
            ztarget.append(float(row[0]))
            utarget.append(float(row[1]))
            nhtarget.append(float(row[2]))
            ntarget.append(float(row[3]))
            rtarget.append(float(row[4]))
            agntarget.append(float(row[5]))
            training.append(row[6:])
        except:
            names.append(row[6:])
        
file_reader.close()       
  
############################# PREPROCESS DATA #############################

[names] = names

#convert training to float 
#convert values to float 
for i in range(len(training)):
    for j in range(len(training[0])):
        training[i][j] = float(training[i][j])

#normalize data using zscores
trans = matrixTranspose(training)
newTraining = []
 
for i in range(len(trans)):  
    Min = min(trans[i])
    Max = max(trans[i])
    for j in range(len(trans[0])):
        try: 
            trans[i][j] = (trans[i][j] - Min)/ (Max - Min)
        except: 
            pass
    item = boxcox(trans[i])
    newTraining.append(item)


training2 = matrixTranspose(newTraining)

#remove unuseful data
training3, names2 = removeData(training2, names)

print('cross validation')
#print('Z')
#cross_validation(5, training3, ztarget,names2) #(binary) #0.045
#print('U')
#cross_validation(5, training3, utarget,names2) #11.24
#print('R')
#cross_validation(5, training3, rtarget,names2) #0.065
#print('n')
#cross_validation(5, training3, ntarget,names2) #(binary) 101145.89
#print('nh')
#cross_validation(5, training3, nhtarget,names2) #0.28
#print('agn')
#cross_validation(5, training3, logFunction(agntarget),names2) #664.87


"""


#read file 
training = []
names = []
utarget = []
ztarget = []
ntarget = []
nhtarget = []
agntarget = [] 
rtarget = []
redshift = []

file_reader = open('colorgridoutput6.csv', "r")
read = csv.reader(file_reader)
for row in read:
    if(row[3] != ''):
       
        try: 
            ztarget.append(float(row[0]))
            utarget.append(float(row[1]))
            nhtarget.append(float(row[2]))
            ntarget.append(float(row[3]))
            rtarget.append(float(row[4]))
            agntarget.append(float(row[5]))
            redshift.append(float(row[6]))
            training.append(row[15:])
        except:
            names.append(row[15:])
        
file_reader.close()   


training2 = []
agntarget2 = [] 

for i in range(len(ztarget)):
    if (ztarget[i] == 0.1 and redshift[i] == 0 and ntarget[i] == 300 and nhtarget[i] == 21 and rtarget[i] == 21.291):
        training2.append(training[i])
        agntarget2.append(agntarget[i])

#training3 = matrixTranspose(training2)
trainingLog = []
for i in training2:
    trainingLog.append(logFunction(i))
    
print(trainingLog)
print('agn')
cross_validation(5, trainingLog, logFunction(agntarget2 ,names) #664.87


