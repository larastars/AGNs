#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:19:24 2019

@author: larakamal
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import math
import csv

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
#convert to log function 
def logFunction(list):
    list2 = toFloat(list)
    result = []
    for i in list2:
        ans = 0.0
        if (i != 0.0):
            ans = math.log(i,10)
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

def plotSingle(x,y,xname,yname):
     fig, ax = plt.subplots()
    
     ax.plot(x, y,'o')
        
     ax.set(ylabel= yname,xlabel=xname)
     plt.title(xname + ' vs. '+ yname)
        
    #ax.grid()
     plt.show()
     rcParams.update({'figure.autolayout': True})
     fileName = 'agnplotsUrestricted/' + (yname+ ' ' +xname).replace('%','').replace('.','')
    
     #print(fileName)
     fig.savefig(fileName)   # save the figure to file
     
     plt.close(fig)    # close the figure

def contourPlot(x,y,z,xname,yname,zname):
    plt.figure()
    x1, y1 = np.meshgrid(x, y)
    cp = plt.contourf(x, y, z)
    plt.colorbar(cp)
    plt.title(zname)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()   
    
def scatterMap(x,y,z,xname,yname,zname,filename):
    fig, ax = plt.subplots(figsize=(7,5))

    plt.title(zname)
    plt.xlabel(xname)
    plt.ylabel(yname)
    
    marker_size=40
    plt.scatter(x, y, marker_size, c=z)
    cbar= plt.colorbar()
    cbar.set_label(zname, labelpad=+1)
    plt.show()
    
    fileName = filename + '/' + 'scatter ' + (xname).replace('/','').replace('.','') + (yname).replace('/','').replace('.','')
    print(fileName)
    fig.savefig(fileName)   # save the figure to file

def div(a,b):
    if b == 0:
        return 0
    else:
        return (a/b)
    
def paperPlots(x,y,z,xname,yname,zname):
    #make sure that x,y,z,x2,y2 are floats
    fig, ax = plt.subplots(figsize=(8,8))
 
    print('before plotting')
    for i in range(len(z)):
        if (x[i] != 0 and y[i] !=0):
            print(i)
       # if (z[i] == 0):
        #for color in ['r', 'b', 'g', 'k', 'm', 'c','y','w']:
           # plt.plot(div(x[i],x2[i]), div(y[i],y2[i]), 'ro', color = 'r')
        #   plt.plot(x[i], y[i], 'ro', color = 'r')
            if (z[i] == 5): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]), 'ro', color = 'b')
                plt.plot(x[i], y[i], 'ro', color = 'b')
            elif(z[i] == 10): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]),'ro', color = 'g')
                plt.plot(x[i], y[i], 'ro', color = 'g')
            elif(z[i] == 20): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]), 'ro', color = 'k')
                plt.plot(x[i], y[i], 'ro', color = 'k')
            elif(z[i] == 40): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]),'ro', color = 'm')
                plt.plot(x[i], y[i], 'ro', color = 'm')
            elif(z[i] == 60): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]),'ro', color = 'c')
                plt.plot(x[i], y[i], 'ro', color = 'c')
            elif(z[i] == 80): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]),'ro', color = 'y')
                plt.plot(x[i], y[i], 'ro', color = 'y')
            elif(z[i] == 100): 
            #plt.plot(div(x[i],x2[i]), div(y[i],y2[i]),'ro', color = 'tab:orange')
                plt.plot(x[i], y[i], 'ro', color = 'tab:orange')
            
    print('done plotting')
    #plt.ylim([1,9])
    #plt.xlim([1,9]) 
    #r_patch = mpatches.Patch(color='r', label='0% AGN')
    b_patch = mpatches.Patch(color='b', label='5% AGN')
    g_patch = mpatches.Patch(color='g', label='10% AGN')
    k_patch = mpatches.Patch(color='k', label='20% AGN')
    m_patch = mpatches.Patch(color='m', label='40% AGN')
    c_patch = mpatches.Patch(color='c', label='60% AGN')
    y_patch = mpatches.Patch(color='y', label='80% AGN')
    w_patch = mpatches.Patch(color='tab:orange', label='100% AGN')

    print('done labeling')
    plt.legend(handles=[b_patch, g_patch, k_patch, m_patch, c_patch, y_patch, w_patch], loc='lower right')
    xlabel = xname 
    ylabel = yname 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(zname)
    plt.show()
    #ax.legend(loc='upper left')
    filename = 'paperplots'
    #plt.show()   
    
    print('before saving')
    fileName = filename + '/' + (xlabel).replace('/','').replace('.','') +  (ylabel).replace('/','').replace('.','') +zname.replace('.','')
        #print(fileName)
    fig.savefig(fileName)   # save the figure to file
         
    plt.close(fig)    # close the figure
    
############################# READ TRAINING DATA #############################
        
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

#open a new file  and store the data in a list 
file_reader = open('colorgridoutput6.csv', "r")
read = csv.reader(file_reader)
for row in read:
    if(row[3] != ''):
        #adding the information to a list 
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



############################# Z = 1 #############################
#Z==1 & redshift==0 & n==300 & NH=21 & R==21.291
"""
training2 = []
utarget2 = []
agntarget2 = [] 

for i in range(len(ztarget)):
    if (ztarget[i] == 1 and redshift[i] == 0 and ntarget[i] == 300 and nhtarget[i] == 21 and rtarget[i] == 21.291):
        utarget2.append(utarget[i])
        training2.append(training[i])
        agntarget2.append(agntarget[i])

training3 = matrixTranspose(training2) 

#plot emission line vs agntarget2        
#for i in range(len(training3)):
#    plotSingle(agntarget2, training3[i], 'AGN %', names[0][i])

#plot log emission line vs agntarget2        
for i in range(len(training3)):
   # curr = logFunction(training3[i])
   # plotSingle(agntarget, curr , 'AGN %', 'Log (' + names[0][i] +')')
    plotSingle(agntarget2, toFloat(training3[i]), 'AGN %', names[0][i] )


#countor plot 
#format [names, [] ]
Z1index = [1,20,25,26,27,31,32,35,36,37,38,39,40,41,67,68,89,91,104]
Z0p1index = [11,20,25,26,27,28,32,34,37,38,39,40,41,67,91,99,103]

Z1training = []
Z0p1training = []


#fill list
for i in Z1index:
    Z1training.append([names[0][i], training3[i]])


for i in Z0p1index:
    Z0p1training.append([names[0][i], training3[i]])
    
 
for i in range(len(Z1training)):
    for j in range(len(Z1training)):
        if (i != j):
            #contourPlot(Z1training[i][1],Z1training[j][1],agntarget2,Z1training[i][0],Z1training[j][0],'% AGN')
            scatterMap(logFunction(Z1training[i][1]),logFunction(Z1training[j][1]),agntarget2,'Log (' + Z1training[i][0] + ')', 'Log (' +Z1training[j][0] + ')','% AGN','scatterZ1')


for i in range(len(Z0p1training)):
    for j in range(len(Z0p1training)):
        if (i != j):
            #contourPlot(Z1training[i][1],Z1training[j][1],agntarget2,Z1training[i][0],Z1training[j][0],'% AGN')
            scatterMap(logFunction(Z0p1training[i][1]),logFunction(Z0p1training[j][1]),agntarget2,'Log (' + Z0p1training[i][0]+ ')','Log (' + Z0p1training[j][0]+ ')','% AGN','scatterZ0p1')



"""

######################### paperplots ############################

[names2] =names


#find u subset 
trainingResult2 = []
agntarget2 = []
ztarget2 = []
utarget2 = []
nhtarget2 = []
ntarget2 = []
rtarget2 = []
redshift2 = []

for i in range(len(utarget)):
    #restricting the U value 
    if (float(utarget[i]) <= 0.01 and float(utarget[i]) >= 0.001):
        if (float(redshift[i]) == 0): 
            if(float(ntarget[i]) == 300):
                if (float(ztarget[i]) == 1.0):
                    if (float(nhtarget[i]) == 21): #**
                      
                    #nhtarget[i] == 21 and rtarget[i] == 21.291
                            trainingResult2.append(training[i])
                            agntarget2.append(agntarget[i])
                            ztarget2.append(ztarget[i])
                            utarget2.append(utarget[i])
                            nhtarget2.append(nhtarget[i])
                            ntarget2.append(ntarget[i])
                            rtarget2.append(rtarget[i])
                            redshift2.append(redshift[i])

training2 = matrixTranspose(trainingResult2)

#for i in range(len(training2)):
#    plotSingle(toFloat(agntarget2), toFloat(training2[i]), 'AGN %', names[0][i])
#print(names2)
a = 'K6_5.573m'
c = 'K6_8.821m'
b = 'SI7_2.481m'
d = 'NA6_14.40m'

#a = 'SI7_2.481m'
#a = 'K6_5.573m'
#c = 'MG7_9.033m'
#b = 'NA6_14.40m'
#e = 'FE7_2.629m'

temp = []
for i in range(len(training2[0])):
    temp.append(1.0)
    
#training2 = training2[:20000]
#agntarget = agntarget[:20000]
xVal = []
yVal = []

curr1 = toFloat(training2[names2.index(a)]) 
curr2 = toFloat(training2[names2.index(b)])
curr3 = toFloat(training2[names2.index(c)])
curr4 = toFloat(training2[names2.index(d)])


for i in range(len(curr1)):
    val1 = 0.0
    val2 = 0.0
    if (curr2[i] != 0.0):    
        val1 = curr1[i] / curr2[i]
   
    if (curr4[i] != 0.0):    
        val2 = curr3[i] / curr4[i]
        
    xVal.append(val1)
    yVal.append(val2)
    

#print(logFunction(yVal))
xname = 'Log(' + a + ' / ' + b + ')'
yname = 'Log(' + c + ' / ' + d + ')'
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(agntarget2), xname, yname,'AGN %')
#paperPlots(x,x2,y,y2,z,xname,xname2,yname,yname2,zname):
#{'K6_5.573m', 'K6_8.821m', 'SI7_2.481m', 'NA6_14.40m'}
#{'SI7_2.481m', 'K6_5.573m', 'MG7_9.033m', 'NA6_14.40m', 'FE7_2.629m'}



