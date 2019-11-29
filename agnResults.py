#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:32:18 2019

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
import warnings
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')


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

#convert to log function 
def logFunction(list):
    list2 = toFloat(list)
    result = []
    for i in list2:
        try:
            ans = round(math.log(i,5),3)
        except:
            ans = 0.0
        result.append(ans)
        #time.sleep(0.1)
    return result

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

#convert a list to float 
def toString(list):
    list2 = []
    for i in list:
        try:
            list2.append(str(i))
        except:
            #print('error',i)
            [ans] = i
            list2.append(str(ans))
    return list2
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


def sort(a,b,c,d,e,f,g,h):
    index = []
    curr = []
    for i in a:
        curr.append(i)
        
    curr.sort()
    
    
    #count the indicies 
    for i in range(len(a)):
        ele = a.index(curr[i]) + b.index(curr[i])  + c.index(curr[i]) + d.index(curr[i]) + e.index(curr[i]) + f.index(curr[i]) + g.index(curr[i])+ h.index(curr[i])
        index.append(ele)
        
    #sort curr based on index, from low to high
    result = sorted(curr, key = lambda x: index[curr.index(x)])
    return result


def boxcox(x):
    xt =[]
    try:
        xt, maxlog, interval = stats.boxcox(x, alpha=0.05)
    except:
        for i in range(len(x)):
            if (x[i] == 0):
                x[i]=0.0000000001
                #x[i]=np.nextafter(0, 1)
        xt, maxlog, interval = stats.boxcox(x, alpha=0.05)
    return xt     


def lassoReg(X,y,names):
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    #coef = pd.Series(reg.coef_, index = X.columns)
    print("Lasso picked " + str(np.sum(reg.coef_ !=0)) + " variables and eliminated the other " +  str(np.sum(reg.coef_ ==0)) + " variables")
    
    
def cross_validation(k,training,target,names,alpha):
    fold = 100/k
    fold = fold/100
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #larger alpha = less coef 
    lasso = Lasso(alpha=alpha)
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
    result = []
    for i in elementList2:
        result.append(i[0])
    
    #find mse
    #mse = mean_squared_error(toFloat(y_test), toFloat(y_score))
    mse = lasso.score(x_train,y_train)
    print(result)
    print('# of coeff = ', nonzero)
    print('mse = ', mse)
    print('')
    print('')
    
    return result
        
    
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


#find u subset 
trainingResult2 = []
agntarget2 = []
ztarget2 = []
utarget2 = []
nhtarget2 = []
ntarget2 = []
rtarget2 = []
#redshift2 = []

for i in range(len(utarget)):
    
    #restricting the U value 
    if (float(utarget[i]) <= 0.01 and float(utarget[i]) >= 0.001):
     
        #if (float(redshift[i]) == 0): 
           # if(float(ntarget[i]) == 300):
                #if (float(ztarget[i]) == 1.0):
                 #   if (float(nhtarget[i]) == 21): #**
                      
                    #nhtarget[i] == 21 and rtarget[i] == 21.291
                            trainingResult2.append(trainingRem[i])
                            agntarget2.append(agntarget[i])
                            ztarget2.append(ztarget[i])
                            utarget2.append(utarget[i])
                            nhtarget2.append(nhtarget[i])
                            ntarget2.append(ntarget[i])
                            rtarget2.append(rtarget[i])
                         #   redshift2.append(redshift[i])

trainingTrans = matrixTranspose(trainingResult2)

"""
#select a list of 0% agn = 0 but 10% !=0
agn_activity_index = []   
for i in range(len(trainingTrans)):
    flag = True
    for j in range(len(trainingTrans[i])):
        if (float(trainingTrans[i][j]) != 0 and float(agntarget[j]) == 0):
            flag = False
    if (flag):
        agn_activity_index.append(i)
"""
#select a list of 0% agn != 0 
agn_activity_index = []   
for i in range(len(trainingTrans)):
    flag = False
    for j in range(len(trainingTrans[i])):
        if (float(trainingTrans[i][j]) != 0 and float(agntarget2[j]) == 0):
            flag = True
    if (flag):
        agn_activity_index.append(i)


#find brightest agn
agn0 = []
agn5 = []
agn10 = []
agn20 = []
agn40 = []
agn60 = []
agn80 = []
agn100 = []


#rank based on the brightest 
for i in agn_activity_index: 
    sumVar0 = 0.0
    sumVar5 = 0.0
    sumVar10 = 0.0
    sumVar20 = 0.0
    sumVar40 = 0.0
    sumVar60 = 0.0
    sumVar80 = 0.0
    sumVar100 = 0.0
    
    for j in range(len(trainingTrans[i])):
        if (float(agntarget2[j]) == 0):
            sumVar0 = sumVar0 + float(trainingTrans[i][j])
        elif (float(agntarget2[j]) == 5):
            sumVar5 = sumVar5 + float(trainingTrans[i][j])
        elif (float(agntarget2[j]) == 10):
            sumVar10 = sumVar10 + float(trainingTrans[i][j])
        elif (float(agntarget2[j]) == 20):
            sumVar20 = sumVar20 + float(trainingTrans[i][j])  
        elif (float(agntarget2[j]) == 40):
            sumVar40 = sumVar40 + float(trainingTrans[i][j])  
        elif (float(agntarget2[j]) == 60):
            sumVar60 = sumVar60 + float(trainingTrans[i][j])
        elif (float(agntarget2[j]) == 80):
            sumVar80 = sumVar80 + float(trainingTrans[i][j])
        elif (float(agntarget2[j]) == 100):
            sumVar100 = sumVar100 + float(trainingTrans[i][j])
            
    agn0.append(sumVar0)
    agn5.append(sumVar5)
    agn10.append(sumVar10)
    agn20.append(sumVar20)
    agn40.append(sumVar40)
    agn60.append(sumVar60)
    agn80.append(sumVar80)
    agn100.append(sumVar100)
            
    
#sorted indecies based on the agn   
agn0 = sorted(agn_activity_index, key = lambda x: agn0[agn_activity_index.index(x)], reverse=True) 
agn5 = sorted(agn_activity_index, key = lambda x: agn5[agn_activity_index.index(x)], reverse=True) 
agn10 = sorted(agn_activity_index, key = lambda x: agn10[agn_activity_index.index(x)], reverse=True) 
agn20 = sorted(agn_activity_index, key = lambda x: agn20[agn_activity_index.index(x)], reverse=True) 
agn40 = sorted(agn_activity_index, key = lambda x: agn40[agn_activity_index.index(x)], reverse=True) 
agn60 = sorted(agn_activity_index, key = lambda x: agn60[agn_activity_index.index(x)], reverse=True) 
agn80 = sorted(agn_activity_index, key = lambda x: agn80[agn_activity_index.index(x)], reverse=True) 
agn100 = sorted(agn_activity_index, key = lambda x: agn100[agn_activity_index.index(x)], reverse=True) 


#ranking from the birghtest to dimest 
result = sort(agn0,agn5,agn10,agn20,agn40,agn60,agn80,agn100)

namesFinal = []
for i in result:
    namesFinal.append(namesRem[i])
#create the title header 
titles = ['agn%', 'U','Z','NH','N','R','redshift']
for i in result:
    titles.append(namesRem[i])
print("ranking from brightest to dimmest")
print(namesFinal)           
  
trainingSet = []
#trans 
for i in result:
    trainingSet.append(toString(trainingTrans[i]))

trainingResult3 = matrixTranspose(trainingSet)

        
#write to file
with open('subset4.csv', mode='w') as file:
        outputwriter = csv.writer(file, delimiter=',')
        outputwriter.writerow(titles)
       
        for i in range(len(ztarget2)):
            curr = [str(agntarget2[i]),str(utarget2[i]),str(ztarget2[i]), str(nhtarget2[i]), str(ntarget2[i]), str(rtarget2[i])]
            for j in range(len(trainingResult3[0])):
                curr.append(str(trainingResult3[i][j]))
                
            outputwriter.writerow(curr)
                                     
file.close()


    


######################### LASSO ########################
#convert to float 
for i in range(len(trainingResult3)):
    for j in range(len(trainingResult3[0])):
        trainingResult3[i][j] = float(trainingResult3[i][j]) 
        

 
print('')
print("consider the top 25")
cut = 50
trainingResultTrans = matrixTranspose(trainingResult3)

#trans 
trainingResult4 = matrixTranspose(trainingResultTrans)

print('')

#larger the # -> less variables 
print('agn')
agn = cross_validation(5,trainingResult4,agntarget2,namesFinal,0.00005)
print('u')
u = cross_validation(5,trainingResult4,utarget2,namesFinal,0.000001)
print('z')
z = cross_validation(5,trainingResult4,ztarget2,namesFinal,0.00001)
print('r')
r = cross_validation(5,trainingResult4,rtarget2,namesFinal,0.0001)
print('n')
n = cross_validation(5,trainingResult4,ntarget2,namesFinal,0.09)
print('nh')
nh = cross_validation(5,trainingResult4,nhtarget2,namesFinal,0.0001)

totalList = u + z +r + n + nh
output = set(agn) - set(totalList)
print('Using top 25, select the ones that are most sensetive to AGN but least sensetive to other parameters using LASSO Regression')
print(output)


