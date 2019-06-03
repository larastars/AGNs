#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:11:52 2019

@author: larakamal
"""

import csv 
import matplotlib
import matplotlib.pyplot as plt
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
        
        """
        ################ plot target ##################
       
        #convert to float and string 
        trainingInst = toFloat(trainingTrans[i])
        target2 = logFunction(target)
        
        #plot 
        ax.plot(target2, trainingInst,'go')
        

        ax.set(ylabel= names[i],xlabel='Log(Mass of blackhole)')
        plt.title(names[i] +' vs. Log(Mass of blackhole)')
        
        ax.grid()
        plt.show()
        rcParams.update({'figure.autolayout': True})
        fileName = 'plotstest/' +(names[i].replace('/','').replace('01','').replace(')',''))
        
        """ 
        ###############################################
        
        
        ################ plot utarget ##################    
        
         #convert to float and string 
        trainingInst = toFloat(trainingTrans[i])
        
        #plot 
        ax.plot(target, trainingInst,'go')
        

        ax.set(ylabel= names[i],xlabel='Ionization Parameters')
        plt.title(names[i] +' vs. Ionization Parameters')
        
        ax.grid()
        plt.show()
        rcParams.update({'figure.autolayout': True})
        fileName = 'plotsutest/' +(names[i].replace('/','').replace('01','').replace(')',''))
       
        ###############################################
        
        fig.savefig(fileName)   # save the figure to file
        plt.close(fig)    # close the figure
    

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

def plotSingle(training, target, names, i):
     fig, ax = plt.subplots()
    
    #convert to float and string 
     trainingInst = toFloat(training)
     #target2 = logFunction(target)
       
    #plot 
     ax.plot(target, trainingInst,'o')
        
     ax.set(ylabel= names[i],xlabel='Log(Mass of blackhole)')
     plt.title(names[i] +' vs. Log(Mass of blackhole) ' + str(i))
        
     ax.grid()
     plt.show()
     rcParams.update({'figure.autolayout': True})

def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    #convert training to float
    trainingFloat = []
    for i in training:
        trainingFloat.append(toFloat(i))
        
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #logistic regression 
    lm = LinearRegression()
    #lm.fit(trainingFloat,target)
    
    #rfe = RFE(model,k)
   # fit = rfe.fit(x_train, y_train)
    
    lm.fit(x_train, y_train)
    
    #test
    y_score = lm.predict(x_test)
    
    mse = mean_squared_error(toFloat(y_test), toFloat(y_score))
    print(mse)

    print(lm.feature_importance_)
    
    #print(y_test)
   # print(y_score)
    #y_test = ''.join(y_test)
    #y_score = ''.join(y_score)
    #print('scores:')
    #print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
    #print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
    #print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
    #print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
    #print(' ')
    
    
def applyRegression(training, target, names):
  
    trainingTrans =  matrixTranspose(training) 
    trainingFloat = []
    elementList = []
    
    #convert training to float
    for i in training:
        trainingFloat.append(toFloat(i))
        
    #create model and fit model 
    lm = LinearRegression()
    lm.fit(trainingFloat,target)
        
    coef = lm.coef_
    #inter = lm.intercept_ 
        
    #print('Score: ', score)
    #print('Coefficients: ', coef)
    #print('Intercept: ', inter)
        
   
    #mse = mean_squared_error(trainingFloat[0], target)
    
    #print("Mean squared error: %.2f" %mse )
    #plotSingle(trainingInst2, target2, names, i) 
    for i in range(len(names)):
        current = []
        current.append(names[i])
        current.append(coef[i])
        
        #mse 
        y_score = lm.predict(training)
        mse = mean_squared_error(toFloat(target), toFloat(y_score))
        #print(mse)
        
        #current.append(mse)
        #current.append(mse)
        elementList.append(current)
  
    #[name, coef, mse]
    return elementList


def normalize(list1, list2, list3):
    #of the form [name, coef, mse]
    listN1= []
    listN2= []
    listN3= []
    coefList = []
    mseList = []
    
    #list1, 2 and 3 have the same length 
    for i in range(len(list1)):
        coefList.append(list1[i][1])
        coefList.append(list2[i][1])
        coefList.append(list3[i][1])
        mseList.append(list1[i][2])
        mseList.append(list2[i][2])
        mseList.append(list3[i][2])
    
    minCoef=min(coefList)
    maxCoef=max(coefList)
    minMse=min(mseList)
    maxMse=max(mseList)
    
    for i in range(len(list1)):
        coef = list1[i][1] - minCoef / (maxCoef-minCoef)
        mse = list1[i][2] - minMse / (minMse - maxMse)
        curr = [list1[i][0], coef, mse]
        listN1.append(curr)
        
        coef = list2[i][1] - minCoef / (maxCoef-minCoef)
        mse = list2[i][2] - minMse / (minMse - maxMse)
        curr = [list2[i][0], coef, mse]
        listN2.append(curr)
        
        coef = list3[i][1] - minCoef / (maxCoef-minCoef)
        mse = list3[i][2] - minMse / (minMse - maxMse)
        curr = [list3[i][0], coef, mse]
        listN3.append(curr)
    
    return listN1, listN2, listN3

def getResult(MassBin, uListBin):
    list = []
    for i in range(len(MassBin)):
        for j in range(len(uListBin)):
            if (MassBin[i][0] == uListBin[j][0]):
                curr = []
                curr.append(MassBin[i][0])
                final = abs(MassBin[i][1]) -MassBin[i][2] - abs(uListBin[j][1]) -uListBin[j][2]
                curr.append(final)
                #curr.append(MassBin[i][1])
                #curr.append(MassBin[i][2])
                #curr.append(uListBin[i][1])
                #curr.append(uListBin[i][2])
               
                list.append(curr)
    list.sort(key=lambda x: x[1], reverse =True)
    return list

def printToFile(list1, list2, list3):

        
    with open('output.csv', mode='w') as file:
        outputwriter = csv.writer(file, delimiter=',')
        outputwriter.writerow(['Low Mass AGNs', 'Medium Mass AGNs','High Mass AGNs'])
        for i in range(len(list1)):
            outputwriter.writerow([str(list1[i]), str(list2[i]),str(list3[i])])
    file.close()
    
def findSubset(training, target, utarget, names):
   
    #break into bins
    trainingbin1, targetbin1, targetubin1, trainingbin2, targetbin2, targetubin2, trainingbin3, targetbin3, targetubin3 = breakInBins(training, target, utarget)
    
    
    MassBin1 = applyRegression(trainingbin1, targetbin1, names2)
    #MassBin2 = applyRegression(trainingbin2, targetbin2, names2)
    #MassBin3 = applyRegression(trainingbin3, targetbin3, names2)
    #uListBin1 = applyRegression(trainingbin1, targetubin1, names2)
    #uListBin2 = applyRegression(trainingbin2, targetubin2, names2)
    #uListBin3 = applyRegression(trainingbin3, targetubin3, names2)
    
    
    #[name, coef, mse]
    #mass bin: high |coef| low mse -> maximize |coef| - mse mass 
    #u bin: low |coef| low mse -> - mse u - |coef|
   # list1 = getResult(MassBin1, uListBin1)
   # list2 = getResult(MassBin2, uListBin2)
   # list3 = getResult(MassBin3, uListBin3)
    #has [name, final, mass coef, mass mse, u coef, u mse]
   # printToFile(list1,list2,list3) 
   #return list1,list2,list3
    return [], [], []
    
    
def createGraph(list1, list2, list3, x, y):
    px = []
    py = []
    for i in list1:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
            
    for i in list2:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
            
    for i in list3:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
   
    #for color in ['r', 'b', 'g', 'k', 'm']:
    plt.plot(px[0], py[0], 'ro', color = 'r', label = 'Low Mass AGNs')
    plt.plot(px[1], py[1], 'ro', color = 'b', label = 'Medium Mass AGNs')
    plt.plot(px[2], py[2], 'ro', color = 'g', label = 'High Mass AGNs')

      
    plt.legend(loc='upper left')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Score Plot')
        
    
    plt.show()   
  
def createDataGraph(list1, list2, list3, training2, names2, target, x, y):
    px = []
    py = []
    for i in list1:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
            
    for i in list2:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
            
    for i in list3:
        if (i[0] == x):
            px.append(float(i[1]))
        if (i[0] == y):
            py.append(float(i[1]))
   
    #for color in ['r', 'b', 'g', 'k', 'm']:
    plt.plot(px[0], py[0], 'ro', color = 'r', label = 'Low Mass AGNs')
    plt.plot(px[1], py[1], 'ro', color = 'b', label = 'Medium Mass AGNs')
    plt.plot(px[2], py[2], 'ro', color = 'g', label = 'High Mass AGNs')

    indexx = 0
    indexy = 0
    #find index
    for i in range(len(names)):
        if (names[i] == x):
            indexx = i
        if (names[i] == y):
            indexy = i
    #print data
   # for i in range(len(training)):
     #   plt.plot(training[i][indexx], training[i][indexy], 'o', color = 'k')
    #plt.plot(training[0][indexx], training[0][indexy], 'o', color = 'k')   
   
      
    plt.legend(loc='upper left')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Score Plot')
        
    
    plt.show()  
    
############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
utarget= []
file_reader = open('RatiosGrid_test3.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[:1] != '' and row[3:][0] != ''):
        utarget.append(row[1:2])
        target.append(row[:1])
        training.append(row[3:])
        
#remove the labelling row 
[names] = training[:1]
training = training[1:]

target = target[1:] 
utarget = utarget[1:]    


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
    utarget[i] = utarget[i][0]
    
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
    zscoreList = stats.zscore(i)
    zscoreList2 = []
    for j in zscoreList:
        num = str(round(j,3))
       # print(num)
        if (num == 'nan'):
            num2 =0
            zscoreList2.append(num2)
        else:
            num2 = float(num) 
            zscoreList2.append(num2)
        
    newTraining.append(zscoreList2)

training2 = matrixTranspose(newTraining)

#remove unuseful data
training3, names2 = removeData(training2, names)

cross_validation(5,training3,targetlog)
#list1,list2,list3  = findSubset(training3, targetlog, utarget, names2)    


#createGraph(list1,list2,list3,'S9(1)/S3(18)', 'S9(3)/S3(18)')
#createGraph(list1,list2,list3,'Na4(9)/Na3', 'Na4(21)/Na3')

#low mass 
print('low mass')
##createGraph(list1,list2,list3,'Na6(8)/Na4)', 'Na6(8)/Na4(6)')
#createGraph(list1,list2,list3,'Na6(8)/Na4(21)', 'Na6(14)/Na4(6)')
#createGraph(list1,list2,list3,'Na6(14)/Na4(21)', 'Al6(9)/Al5')

#medium mass

print('medium mass')
#createGraph(list1,list2,list3,'Si11/Si10', 'Si7(6)/Si6')
#createGraph(list1,list2,list3,'Fe13/Fe6(1.01)', 'Si11/Si6')
#createGraph(list1,list2,list3,'Si9/Si6', 'Al8(5)/Al6(9)')

#high mass 
print('high mass')
#createGraph(list1,list2,list3,'Si11/Si10', 'Si11/Si6')
#createGraph(list1,list2,list3,'Si11/Si7(2)', 'Si11/Si7(6)')
#createGraph(list1,list2,list3,'Si10/Si6', 'Si9/Si6')




#createDataGraph(list1,list2,list3,MassBin1, MassBin2, MassBin3,'Mg5(5)/Mg4', 'Si10/Si9')

