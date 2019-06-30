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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import rcParams
from scipy import stats
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Lasso, LassoCV
import pandas as pd
import matplotlib.patches as mpatches
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
    

def cross_validation(k,training,target,a):
    fold = 100/k
    fold = fold/100
    mse = 0
    #convert training to float
    trainingFloat = []
    for i in training:
        trainingFloat.append(toFloat(i))
    for i in range(k):
        #split
        x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
        lasso = Lasso(alpha=a, max_iter =46000)
        #0.0007
        #lasso = Lasso(alpha=0.000001, max_iter =46000)
    
        lasso.fit(x_train,y_train)
    
        #test
        y_score = lasso.predict(x_test)
    
        mse += mean_squared_error(toFloat(y_test), toFloat(y_score))
        
    print(np.sum(lasso.coef_ !=0))
    print('validation error = ', mse/k)
    
    """
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(y_score)):
        if (float(y_test[i]) <= 4):
        #for color in ['r', 'b', 'g', 'k', 'm']:
            plt.plot(float(y_test[i]), float(y_score[i]), 'ro', color = 'r')
        elif (float(y_test[i]) > 4 and float(y_test[i]) <= 6):  
            plt.plot(float(y_test[i]), float(y_score[i]),'ro', color = 'b')
        else:

            plt.plot(float(y_test[i]), float(y_score[i]), 'ro', color = 'g')

      
    red_patch = mpatches.Patch(color='r', label='Low Mass AGNs')
    blue_patch = mpatches.Patch(color='b', label='Intermediate Mass AGNs')
    green_patch = mpatches.Patch(color='g', label='High Mass AGNs')

    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.xlabel('Log(Actual AGN Mass)')
    plt.ylabel('Log(Lasso Regression Prediction)')
    plt.title('Log(Actual AGN Mass) vs. Log(Lasso Regression Prediction)')
    #ax.legend(loc='best')
    #filename = 'newplots4'
    plt.show()   
    """
    #fileName = filename + '/' + ('Log(Actual AGN Mass)').replace('/','').replace('1.01','1') +  ('Log(Lasso Regression Prediction) 2').replace('/','').replace('1.01','1')
        #print(fileName)
    #fig.savefig(fileName)   # save the figure to file
         
   # plt.close(fig)    # close the figure

  
def writeEquation(elementList, inst, names):
    
    #remove zeros
    elementList2 = []
    for i in range(len(elementList)):
        if (elementList[i][1] != 0.0):
           elementList2.append(elementList[i])

    equation = ''
    for i in elementList2:
        if (i == elementList2[0] or float(i[1]) < 0):
            equation += ' ' + str(i[1]) + '*' + i[0].replace("/","") 

        else:
            equation += ' + ' + str(i[1]) + '*' + i[0].replace("/","") 
            
    #print(equation)
    instList = []
    for i in range(len(inst)):
        instList.append([names[i], inst[i]])
    
    ans = 0
    for i in elementList2:
        for j in instList:
            if (i[0] == j[0]):
               ans += float(i[1]) * float(j[1]) 
    print(ans)


def solve(training,target,model) :
    coef = model.coef_
    inter = model.intercept_
    ans = []
    for i in training:
        ans.append(round(np.dot(i,coef) + inter,1))
    
   # for i in range(len(ans)):
    #    print(ans[i], '  ', target[i])
    mse = mean_squared_error(toFloat(target), toFloat(ans))
    print('mse = ', mse)
    
    
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(ans)):
        if (float(target[i]) <= 4):
        #for color in ['r', 'b', 'g', 'k', 'm']:
            plt.plot(float(target[i]), float(ans[i]), 'ro', color = 'r')
        elif (float(target[i]) > 4 and float(target[i]) <= 6):  
            plt.plot(float(target[i]), float(ans[i]),'ro', color = 'b')
        else:

            plt.plot(float(target[i]), float(ans[i]), 'ro', color = 'g')

    #plt.ylim([1,9])
    #plt.xlim([1,9]) 
    red_patch = mpatches.Patch(color='r', label='Low Mass AGNs')
    blue_patch = mpatches.Patch(color='b', label='Intermediate Mass AGNs')
    green_patch = mpatches.Patch(color='g', label='High Mass AGNs')

    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper left')
    plt.xlabel('Log(Actual AGN Mass)')
    plt.ylabel('Log(Lasso Regression Prediction)')
    plt.title('Log(Actual AGN Mass) vs. Log(Lasso Regression Prediction)')
    #ax.legend(loc='upper left')
    filename = 'newplots4'
    #plt.show()   
    
    fileName = filename + '/' + ('Log(Actual AGN Mass)').replace('/','').replace('1.01','1') +  ('Log(Lasso Regression Prediction) 0p009').replace('/','').replace('1.01','1')
        #print(fileName)
    fig.savefig(fileName)   # save the figure to file
         
    plt.close(fig)    # close the figure

def solve2(x,y,target,model) :
    coef = model.coef_
    inter = model.intercept_
    ans = []
    for i in x:
        ans.append(round(np.dot(i,coef) + inter,1))
    
   
   # for i in range(len(ans)):
    #    print(ans[i], '  ', target[i])
    mse = mean_squared_error(toFloat(y), toFloat(ans))
    print('mse = ', mse)
    
    
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(ans)):
        if (float(target[i]) <= 4):
        #for color in ['r', 'b', 'g', 'k', 'm']:
            plt.plot(float(y[i]), float(ans[i]), 'ro', color = 'r')
        elif (float(target[i]) > 4 and float(target[i]) <= 6):  
            plt.plot(float(y[i]), float(ans[i]),'ro', color = 'b')
        else:

            plt.plot(float(y[i]), float(ans[i]), 'ro', color = 'g')

      
    red_patch = mpatches.Patch(color='r', label='Low Mass AGNs')
    blue_patch = mpatches.Patch(color='b', label='Intermediate Mass AGNs')
    green_patch = mpatches.Patch(color='g', label='High Mass AGNs')

    plt.ylim([-4.5,-1])
    plt.xlim([-4.5,-1])

    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.xlabel('Ionization Parameter')
    plt.ylabel('Lasso Regression Ionization Parameter Prediction')
    plt.title('Ionization Parameter vs. Lasso Regression Ionization Parameter Prediction')
    #ax.legend(loc='best')
    filename = 'newplots4'
    #plt.show()   
    
    fileName = filename + '/' + ('Ionization Parameter').replace('/','').replace('1.01','1') +  ('Lasso Regression Ionization Parameter Prediction 0p004').replace('/','').replace('1.01','1')
        #print(fileName)
    fig.savefig(fileName)   # save the figure to file
         
    plt.close(fig)    # close the figure

def applyPoly(training, target, names,k):
    fold = 100/k
    fold = fold/100
    
    elementList = []
   
    y = np.array(toFloat(target))
    x = []
    #convert training to float
    for i in training:
        x.append(toFloat(i))
    
    # Alpha (regularization strength) of LASSO regression
    lasso_eps = 0.0001
    lasso_nalpha=20
    lasso_iter=5000
    # Min and max degree of polynomials features to consider
    degree_min = 2
    degree_max = 6
    
    # Test/train split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= fold)

    # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
    for degree in range(degree_min,degree_max+1):
        print(degree)
        model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
        print('fit')
        model.fit(x_train,y_train)
        print('predict')
        test_pred = np.array(model.predict(x_test))
        print('metric')
        #RMSE =np.sqrt(np.sum(np.square(test_pred-y_test)))
        mse = mean_squared_error(test_pred, y_test)
        test_score = model.score(x_test,y_test)     
        print("rmse = ", mse)
        print("score = ", test_score)
    return elementList
 
    
def applyRegression2(training, target, utarget, names):
  
    elementList = []
   
    y = np.array(toFloat(target))
    x = []
    
    #convert training to float
    for i in training:
        x.append(toFloat(i))
    
    
    lasso = Lasso(alpha=0.0007, max_iter =46000)
    lasso.fit(x,y)
    
    weight = np.array(lasso.coef_)
    
        
    for i in range(len(weight)):
        elementList.append([names[i],round(weight[i],3)])
    
    print(np.sum(lasso.coef_ !=0))
    inter = lasso.intercept_
     #test
    y_score = lasso.predict(x)
        
    #heatMapSingle(utarget,target,toFloat(y_score), 'Ionization Parameter', 'Log(Mass of AGN)', 'Lasso Predicted Log(Mass of AGN)', 'randoms')
    
    #solve2(x,y,target,lasso)   
    mse = mean_squared_error(toFloat(y), toFloat(y_score))
    print('training error = ', mse)
    #print(np.array(lasso.coef_))
    #[name,weight,0]
  
    return inter, elementList


def applyRegression(training, target, names, a):
  
    elementList = []
   
    y = np.array(toFloat(target))
    x = []
    
    #convert training to float
    for i in training:
        x.append(toFloat(i))
    
    
    lasso = Lasso(alpha=a, max_iter =46000)
    lasso.fit(x,y)
    
    weight = np.array(lasso.coef_)
    
        
    for i in range(len(weight)):
        elementList.append([names[i],round(weight[i],3)])
    
    print(np.sum(lasso.coef_ !=0))
    inter = lasso.intercept_
     #test
    y_score = lasso.predict(x)
  
    #solve(x,y,lasso)   
    mse = mean_squared_error(toFloat(y), toFloat(y_score))
    print('training error = ', mse)
    #print(np.array(lasso.coef_))
    #[name,weight,0]
  
    return inter, lasso, toFloat(y_score)

def applyRegressionPoly(training, target, names):
  
    elementList = []
   
    y = np.array(toFloat(target))
    x0 = []
    x = []
    
    #convert training to float
    for i in training:
        x0.append(toFloat(i))
        x.append(toFloat(i))
    
    x0 = matrixTranspose(x0)
    x = matrixTranspose(x)
    #################change x
    for i in range(len(x0)):
        curr = []
        for j in range(len(x0[0])):
            curr.append(x0[i][j]**2)   
        x.append(curr)  
       
    for i in range(len(x0)):
        curr = []
        for j in range(len(x0[0])):
            curr.append(x0[i][j]**3)   
        x.append(curr)  
   
    for i in range(len(x0)):
        curr = []
        for j in range(len(x0[0])):
            curr.append(x0[i][j]**4)   
        x.append(curr)  
        
    for i in range(len(x0)):
        curr = []
        for j in range(len(x0[0])):
            curr.append(x0[i][j]**5)   
        x.append(curr)  
    #########################
    
    x = matrixTranspose(x)
   
    lasso = Lasso(alpha=0.00009, max_iter =46000)
    lasso.fit(x,y)
    
    weight = np.array(lasso.coef_)
    inter = lasso.intercept_
    names2 = []
    for i in names:
        names2.append(i)
    for i in names:
        names2.append(str(i +'*2'))
    for i in names:
        names2.append(str(i +'*3'))
    for i in names:
        names2.append(str(i +'*4'))
    for i in names:
        names2.append(str(i +'*5'))
        
    for i in range(len(weight)):
        elementList.append([names2[i],round(weight[i],3)])
    
    print(np.sum(lasso.coef_ !=0))
    
     #test
    y_score = lasso.predict(x)
    
    mse = mean_squared_error(toFloat(y), toFloat(y_score))
    print('training error = ', mse)
    #print(np.array(lasso.coef_))
    #[name,weight,0]
  
    return inter, elementList

    
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
                #x[i]=np.nextafter(0, 1)
        xt, maxlog, interval = stats.boxcox(x, alpha=0.05)
    return xt     
    
def findSubset(training, target, names, alpha):

    inter, model, list1 = applyRegression(training, target, names2, alpha)
    #inter, list1 = applyRegressionPoly(training, target, names2)
    #list1.sort(key=lambda x: x[1], reverse =True)
    
   # writeEquation(list1, training[0], names)

    #printToFile(list1, inter)
    return list1


def findSubset2(training, target, utarget, names):

    inter, list1 = applyRegression2(training, target, utarget, names2)
    #list1 = applyPoly(training, target, names2,5)
    list1.sort(key=lambda x: x[1], reverse =True)
    
   # writeEquation(list1, training[0], names)

    printToFile(list1, inter)
    return list1


def plotInst(x,y,xname,yname,target,title, filename):
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(target)):
        if (float(target[i]) <= 4):
        #for color in ['r', 'b', 'g', 'k', 'm']:
            plt.plot(float(x[i]), float(y[i]), 'ro', color = 'r')
        elif (float(target[i]) > 4 and float(target[i]) <= 6):  
            plt.plot(float(x[i]), float(y[i]), 'ro', color = 'b')
        else:
            plt.plot(float(x[i]), float(y[i]), 'ro', color = 'g')

      
    #plt.legend(loc='upper left')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
        
    
    #plt.show()   
    
    fileName = filename + '/' + (xname).replace('/','').replace('1.01','1') +  (yname).replace('/','').replace('1.01','1')
        #print(fileName)
    fig.savefig(fileName)   # save the figure to file
         
    plt.close(fig)    # close the figure

def plotInst2(x,y,xname,yname,title,filename):
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(x)):
        plt.plot(float(x[i]), float(y[i]), 'ro', color = 'b')
       

    #plt.legend(loc='upper left')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
        
    
    #plt.show()   
    
    fileName = filename + '/' + (xname).replace('/','').replace('1.01','1') +  (yname).replace('/','').replace('1.01','1')
        #print(fileName)
    fig.savefig(fileName)   # save the figure to file
         
    plt.close(fig)    # close the figure

    
def heatMap(x,y,z,xname,yname,zname,filename):
    
    z1 = matrixTranspose(z)
    
    
    #for z2 in z1:
    for j in range(len(z1)):
        i = 0
        z3=[]
        y2=[]
        x2=[]
        while(i < len(x)):
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


    
def heatMapSingle(x,y,z,xname,yname,zname,filename):
    
    #z1 = matrixTranspose(z)
    #print(z1)
    #z2 = []
    
    #z2 =list(dict.fromkeys(z))
    #z3 = pad(z2, 0, len(x))
    for i in range(len(y)):
        y[i] = round(float(y[i]),2)
        
    for i in range(len(x)):
        x[i] = round(float(x[i]),2)
    
    
    #print(x)
    matrix = list(zip(x,y,z))
    
    #for i in range(len(x)):
     #   print(x[i], y[i], z[i])
    
    table = pd.DataFrame(matrix, columns=[xname,yname,zname])
    table = table.drop_duplicates([xname,yname])

    #print(table)
    table = table.pivot(xname,yname,zname)
        #8.5,6
        

    #plt.xlim([1,9]) 
    fig, ax = plt.subplots(figsize=(16,16))


    ax = sns.heatmap(table)

    ax.invert_yaxis()
    ax.set_title(zname)
    fileName = filename + '/' + (zname).replace('/','').replace('1.01','1')
    #print(fileName)
    fig.savefig(fileName)   # save the figure to file
            

def scatterMap(x,y,z,xname,yname,zname,filename):
    fig, ax = plt.subplots(figsize=(10,10))

    marker_size=17
    plt.title(zname, fontsize=20)
    plt.xlabel(xname, fontsize=marker_size)
    plt.ylabel(yname, fontsize=marker_size)
    
    marker_size=17
    plt.scatter(x, y, marker_size, c=z)
    cbar= plt.colorbar()
    cbar.set_label(zname, labelpad=+1,fontsize=marker_size)
    plt.show()
    
    fileName = filename + '/' + 'scatter2 ' + (zname).replace('/','').replace('1.01','1')
    print(fileName)
    fig.savefig(fileName)   # save the figure to file

def indpData(training, names, indpSet):
    trainingTrans = matrixTranspose(training)
    newTrainTrans = []
    for i in range(len(names)):
        if (names[i] in indpSet):
            newTrainTrans.append(trainingTrans[i])
            
    newTrain = matrixTranspose(newTrainTrans)
    return newTrain


def trainModel(training, target, names, a):
    
    elementList = []
    #convert training and target to float
    y = np.array(toFloat(target))
    x = []
    
    for i in training:
        x.append(toFloat(i))
    
    #create lasso regression 
    lasso = Lasso(alpha=a, max_iter =46000)
    lasso.fit(x,y)
    weight = np.array(lasso.coef_)
    
    for i in range(len(weight)):
        if (weight[i] != 0):
            elementList.append([round(weight[i],2), names[i]])
    
    elementList.append([round(lasso.intercept_,2), ''])
    print(elementList)

    print("number of coef = ", np.sum(lasso.coef_ !=0))
    
    
    y_score = lasso.predict(x)  
    
    mse = mean_squared_error(toFloat(y), toFloat(y_score))
    print('training error 1 = ', mse)
    return lasso, toFloat(y_score)

def testingModel(training, target, model):
    coef = model.coef_
    coefList = []
    for i in range(len(coef)):
        if (coef[i] !=0):
            print(names[i], coef[i])
            coefList.append(coef[i])
   
    inter = model.intercept_
    y_score = []
    for i in training:
        y_score.append(round(np.dot(i,coefList) + inter,1))
    
    #y_score = model.predict(training)
    mse = mean_squared_error(toFloat(target), toFloat(y_score))
    print('training error 2 = ', mse)
    
def transform(training, names): 
    newTraining = []
    for i in training:  
        item2 = boxcox(i)
        [item] = preprocessing.normalize([item2])
        newTraining.append(item)
        
    training2, names2 = removeData(newTraining, names)
    return training2, names2

def plotactualvspredicted(actual, predicted, target, xname, yname, title): 
    
    
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(predicted)):
        if (float(target[i]) <= 4):
        #for color in ['r', 'b', 'g', 'k', 'm']:
            plt.plot(float(actual[i]), float(predicted[i]), 'ro', color = 'r')
        elif (float(target[i]) > 4 and float(target[i]) <= 6):  
            plt.plot(float(actual[i]), float(predicted[i]),'ro', color = 'b')
        else:

            plt.plot(float(actual[i]), float(predicted[i]), 'ro', color = 'g')

      
    red_patch = mpatches.Patch(color='r', label='Low Mass AGNs')
    blue_patch = mpatches.Patch(color='b', label='Intermediate Mass AGNs')
    green_patch = mpatches.Patch(color='g', label='High Mass AGNs')

    if (float(actual[0]) > 0):
        plt.ylim([1,9])
        plt.xlim([1,9]) 
    else:
        plt.ylim([-4,-1])
        plt.xlim([-4,-1]) 
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    
    

    plt.xlabel(xname, fontsize = 10)
    plt.ylabel(yname, fontsize= 10)
    plt.title(title, fontsize= 12)
    #ax.legend(loc='best')
    #filename = 'newplots4'
    plt.show()   
    
def printToFile(list1, inter):
    
    list2 = []
    for i in list1:
        if (float(i[1]) != 0):
            list2.append(i)
    with open('2223.csv', mode='w') as file:
        outputwriter = csv.writer(file, delimiter=',')
        outputwriter.writerow(['Result'])
        for i in range(len(list2)):
            outputwriter.writerow([str(list2[i])])
        out = '[intercept, ' + str(round(inter,2)) + ']'
        outputwriter.writerow([out])
    file.close()
    
############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
utarget= []
ntarget= []
file_reader = open('RatiosGrid_test4.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[:1] != ['']):# and row[3:][0] != ''):
        utarget.append(row[1:2])
        ntarget.append(row[2:3])
        target.append(row[:1])
        training.append(row[3:])
        
    
file_reader.close()       
#remove the labelling row 
[names] = training[:1]
training = training[1:]

target = target[1:] 
utarget = utarget[1:]    
ntarget = ntarget[1:]  

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
    
for i in range(len(ntarget)):
    ntarget[i] = round(float(ntarget[i][0]),3)
    
for i in range(len(target)):
    target[i] = target[i][0]

#convert values to float 
for i in range(len(target)):
    target[i] = float(target[i])

#appy log function 
targetlog = logFunction(target)  


#normalize data using zscores
#round the number to 3 decimal place
#trans = matrixTranspose(training)
#newTraining = transform(training)
 

#training2 = matrixTranspose(newTraining)

#remove unuseful data
#training3, names2 = transform(training, names)


training4 = []
training5 = []
training6 = []
targetlog4 = []
targetlog5 = []
targetlog6 = []

for i in range(len(utarget)):
    if (float(utarget[i]) <= -3.0):
        training6.append(training[i])
        targetlog6.append(targetlog[i])
    if (float(utarget[i]) <= -2.0 and float(utarget[i]) > -3.0):
        training5.append(training[i])
        targetlog5.append(targetlog[i])
    if(float(utarget[i]) > -2.0):
        training4.append(training[i])
        targetlog4.append(targetlog[i])


listU = ['Mg7(9)/Mg4', 'Na6(8)/Na3', 'Fe13/Fe6(1.01)', 'Na6(14)/Na4(6)', 'Na6(14)/Na4(9)']
trainingU = indpData(training,names,listU)
trainingUT,_ = transform(trainingU,names)
modelU, euw = trainModel(trainingUT, utarget, listU, 0.00001) 
cross_validation(5,trainingUT,utarget,0.00001)



plotactualvspredicted(utarget, euw, targetlog, 'Actual Ionization Parameter', '1.18 [Mg7(9)/Mg4] +0.54 [Na6(8)/Na3] +2.29 [Fe13/Fe6(1.01)] \n -2.20 [Na6(14)/Na4(6)] +0.81 [Na6(14)/Na4(9)] -2.68', 'Actual Ionization Parameter vs. Lasso Predicted Ionization Parameter') 


list12 = ['Mg7(9)/Mg4', 'Na6(8)/Na3', 'Fe13/Fe6(1.01)', 'Na6(14)/Na4(6)', 'Na6(14)/Na4(9)']
training12 = indpData(training4,names,list12)
training12T,_ = transform(training12,names)
model12, e12w = trainModel(training12T, targetlog4, list12, 0.00001) 
cross_validation(5,training12T,targetlog4,0.00001)


plotactualvspredicted(targetlog4, e12w, targetlog4, 'Log(Actual AGN Mass)', '-3.54 [Mg7(9)/Mg4] +7.43 [Na6(8)/Na3] -19.98 [Fe13/Fe6(1.01)]  \n +8.84 [Na6(14)/Na4(6)] -2.62 [Na6(14)/Na4(9)] +18.08', 'Log(Actual AGN Mass) vs. Log(Lasso Regression Prediction) -1 >= U > -2') 


list23 = ['Mg7(9)/Mg4', 'Na6(8)/Na3', 'Fe13/Fe6(1.01)', 'Na6(14)/Na4(6)', 'Na6(14)/Na4(9)']
training23 = indpData(training5,names,list23)
training23T,_ = transform(training23,names)

model23, e23w = trainModel(training23T, targetlog5, list23, 0.00001) 
#cross_validation(5,training23T,targetlog5,0.00001)
plotactualvspredicted(targetlog5, e23w, targetlog5, 'Log(Actual AGN Mass)', '2.42 [Mg7(9)/Mg4] -7.20 [Na6(8)/Na3] +16.44 [Fe13/Fe6(1.01)] \n -47.29 [Na6(14)/Na4(6)] +25.30 [Na6(14)/Na4(9)] +14.31' , 'Log(Actual AGN Mass) vs. Log(Lasso Regression Prediction) -2 >= U > -3') 


list34 = ['Mg7(9)/Mg4', 'Na6(8)/Na3', 'Fe13/Fe6(1.01)', 'Na6(14)/Na4(6)', 'Na6(14)/Na4(9)']
training34 = indpData(training6,names,list34)
training34T,_ = transform(training34,names)

model34, e34w = trainModel(training34T, targetlog6, list34, 0.00001)
#cross_validation(5,training34T,targetlog6,0.00001)
plotactualvspredicted(targetlog6, e34w, targetlog6, 'Log(Actual AGN Mass)', '-27.76 [Mg7(9)/Mg4] -24.84 [Na6(8)/Na3] +50.51 [Fe13/Fe6(1.01)] \n -104.66 [Na6(14)/Na4(6)] -6.97 [Na6(14)/Na4(9)] -47.33', 'Log(Actual AGN Mass) vs. Log(Lasso Regression Prediction) -3 >= U > -4') 


predictedMass = []
actualTarget = []

for i in range(len(e12w)):
    predictedMass.append(e12w[i])
    actualTarget.append(targetlog4[i])
    
for i in range(len(e23w)):
    predictedMass.append(e23w[i])
    actualTarget.append(targetlog5[i])

for i in range(len(e34w)):
    predictedMass.append(e34w[i])
    actualTarget.append(targetlog6[i])
    
xname = 'Diagnostic I'
yname = 'Diagnostic II'
#xname = '-1>=U>-2, -3.54[Mg7(9)/Mg4]+7.43[Na6(8)/Na3]-19.98[Fe13/Fe6(1.01)]+8.84[Na6(14)/Na4(6)]-2.62[Na6(14)/Na4(9)]+18.08 \n -2>=U>-3, 2.42[Mg7(9)/Mg4]-7.20 [Na6(8)/Na3]+16.44 [Fe13/Fe6(1.01)]-47.29[Na6(14)/Na4(6)]+25.30[Na6(14)/Na4(9)]+14.31\n-3>=U>-4,-27.76[Mg7(9)/Mg4]-24.84[Na6(8)/Na3]+50.51[Fe13/Fe6(1.01)]-104.66[Na6(14)/Na4(6)]-6.97[Na6(14)/Na4(9)]-47.33'
#yname = '1.18 [Mg7(9)/Mg4] +0.54 [Na6(8)/Na3] +2.29 [Fe13/Fe6(1.01)] \n -2.20 [Na6(14)/Na4(6)] +0.81 [Na6(14)/Na4(9)] -2.68'
scatterMap(predictedMass,euw,utarget,xname, yname, 'Ionization Parameter',  'randoms')
scatterMap(predictedMass,euw,actualTarget,xname, yname, 'Log(Mass of AGN)',  'randoms')

#predictM23, e23 = trainModel(training5, targetlog5, names2, 0.00414) 
#predictM34, e34 = trainModel(training6, targetlog6, names2, 0.003) 

#e12 = matrixTranspose(e12w)[0]
#e23 = matrixTranspose(e23)[0]
#e34 = matrixTranspose(e34)[0]

#training, names, set 
#training12 = indpData(training,names,e12)

#transform data
#training12T, _ = transform(training12,names)

#training12TU = []
#training23TU = []
#training34TU = []
#get training data
#for i in range(len(utarget)):
   # if (float(utarget[i]) <= -3.0):
    #    training6.append(training3[i])
    #if (float(utarget[i]) <= -2.0 and float(utarget[i]) > -3.0):
     #   training5.append(training3[i])
    #if(float(utarget[i]) > -2):
    #    training12TU.append(training12T[i])

#test
#testingModel(training12TU, targetlog4, model12)
