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
import numpy as np
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
     fileName = 'zplots/' + (yname+ ' ' +xname).replace('%','').replace('.','')
    
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
    
def paperPlots(x,y,z,t,xname,yname,zname):
    #make sure that x,y,z,x2,y2 are floats
    fig, ax = plt.subplots(figsize=(8,8))
    
    if (zname == 'AGN'):
 
        for i in range(len(z)):
           if (x[i] != 0 and y[i] !=0):
              
               if (z[i] == 0):
                   plt.plot(x[i], y[i], 'ro', color = 'b')
               else:
                   plt.plot(x[i], y[i], 'ro', color = 'r')
          
        b_patch = mpatches.Patch(color='b', label='Star Forming')
        r_patch = mpatches.Patch(color='r', label='AGN')
        
        plt.legend(handles=[b_patch,r_patch],loc='upper left')  
     
    
    
    ######################### z #####################
    elif (zname == 'Z'):
        for i in range(len(z)):
           if (x[i] != 0 and y[i] !=0):
              
               if (z[i] == 0 and t[i] == 0.1):
                  
                   plt.plot(x[i], y[i], 'ro', color = 'b')
               
               elif (z[i] == 0 and t[i] == 1.0):
                   plt.plot(x[i], y[i], 'ro', color = 'r')
                   
               elif (z[i] != 0 and t[i] == 1.0):
                   plt.plot(x[i], y[i], 'ro', color = 'g')
               elif (z[i] != 0 and t[i] == 0.1):
                   plt.plot(x[i], y[i], 'ro', color = 'tab:orange')
     
        b_patch = mpatches.Patch(color='b', label='Star Forming, z = 0.1')
        r_patch = mpatches.Patch(color='r', label='Star Forming, z = 1.0')
        g_patch = mpatches.Patch(color='g', label='AGN, z = 1.0')
        o_patch = mpatches.Patch(color='tab:orange', label='AGN, z = 0.1')
        
        plt.legend(handles=[b_patch,r_patch,g_patch, o_patch],loc='upper left')  
    

    #################### nh #########################
    elif (zname=='NH'):
        for i in range(len(z)):
            if (x[i] != 0 and y[i] !=0):
              
                if (z[i] == 19 or z[i] == 19.5):
                    plt.plot(x[i], y[i], 'ro', color = 'b')
                elif (z[i] == 20 or z[i] == 20.5): 
                    plt.plot(x[i], y[i], 'ro', color = 'tab:orange')
                elif (z[i] == 21 or z[i] == 21.5): 
                    plt.plot(x[i], y[i], 'ro', color = 'g')
                elif (z[i] == 22 or z[i] == 22.5): 
                    plt.plot(x[i], y[i], 'ro', color = 'k')
                elif (z[i] == 23 or z[i] == 23.5): 
                    plt.plot(x[i], y[i], 'ro', color = 'm')
                elif (z[i] == 24): 
                    plt.plot(x[i], y[i], 'ro', color = 'c')
                else:
                    plt.plot(x[i], y[i], 'ro', color = 'y')
    
        b_patch = mpatches.Patch(color='b', label='nh:19 - 19.5')
        o_patch = mpatches.Patch(color='tab:orange', label='nh:20 - 20.5')
        g_patch = mpatches.Patch(color='g', label='nh:21 - 21.5')
        k_patch = mpatches.Patch(color='k', label='nh:22 - 22.5')
        m_patch = mpatches.Patch(color='m', label='nh:23 - 23.5')
        c_patch = mpatches.Patch(color='c', label='nh:24')
        
        plt.legend(handles=[b_patch,o_patch,g_patch,k_patch,m_patch,c_patch])     
     
    #plt.ylim([1,9])
    #plt.xlim([1,9]) 
   
   ######################### r ##########################
    elif (zname == 'R'):
        for i in range(len(z)):
            if (x[i] != 0 and y[i] !=0):
                 if (z[i] < 19.5):
                     plt.plot(x[i], y[i], 'ro', color = 'b')
                 elif (z[i] >= 19.5 and z[i] < 20): 
                     plt.plot(x[i], y[i], 'ro', color = 'tab:orange')
                 elif (z[i] >= 20 and z[i] < 20.5): 
                     plt.plot(x[i], y[i], 'ro', color = 'g')
                 elif (z[i] >= 20.5 and z[i] < 21): 
                     plt.plot(x[i], y[i], 'ro', color = 'k')
                 elif (z[i] >= 21 and z[i] < 21.5): 
                     plt.plot(x[i], y[i], 'ro', color = 'm')
                 else:
                     plt.plot(x[i], y[i], 'ro', color = 'c')
                 
    
        b_patch = mpatches.Patch(color='b', label='r:19.0 - 19.5')
        o_patch = mpatches.Patch(color='tab:orange', label='r:19.5 - 20.0')
        g_patch = mpatches.Patch(color='g', label='r:20.0 - 20.5')
        k_patch = mpatches.Patch(color='k', label='r:20.5 - 21.0')
        m_patch = mpatches.Patch(color='m', label='r:21.0 - 21.5')
        c_patch = mpatches.Patch(color='c', label='r:21.5-22.0')
        
        plt.legend(handles=[b_patch,o_patch,g_patch,k_patch,m_patch,c_patch])    
        
  
    ######################### U ##########################
    elif (zname == 'U'):
        for i in range(len(z)):
            if (x[i] != 0 and y[i] !=0):
                 if (z[i] >= 0.01):
                     plt.plot(x[i], y[i], 'ro', color = 'b')
                 elif (z[i] < 0.01 and z[i] >= 0.0082): 
                     plt.plot(x[i], y[i], 'ro', color = 'tab:orange')
                 elif (z[i] < 0.0082 and z[i] >= 0.0064): 
                     plt.plot(x[i], y[i], 'ro', color = 'g')
                 elif (z[i] < 0.0064 and z[i] >= 0.0046): 
                     plt.plot(x[i], y[i], 'ro', color = 'k')
                 elif (z[i] < 0.0046 and z[i] >= 0.0028): 
                     plt.plot(x[i], y[i], 'ro', color = 'm')
                 else:
                     plt.plot(x[i], y[i], 'ro', color = 'c')
                 
    
        #b_patch = mpatches.Patch(color='b', label='U:0.01')
        o_patch = mpatches.Patch(color='tab:orange', label='U:0.01 - 0.0082')
        g_patch = mpatches.Patch(color='g', label='U:0.0082 - 0.0064')
        k_patch = mpatches.Patch(color='k', label='U:0.0064 - 0.0046')
        m_patch = mpatches.Patch(color='m', label='U:0.0046 - 0.0028')
        c_patch = mpatches.Patch(color='c', label='U:0.0028 - 0.001')
        
        plt.legend(handles=[o_patch,g_patch,k_patch,m_patch,c_patch])   
     

    

   # plt.legend(handles=[b_patch, g_patch, k_patch, m_patch, c_patch, y_patch, w_patch], loc='lower right')
    xlabel = xname 
    ylabel = yname 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(zname)
    plt.show()
    #ax.legend(loc='upper left')
    #ax.legend(loc='lower left')
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
#redshift2 = []
for i in range(len(utarget)):
    #restricting the U value 
    if (float(utarget[i]) <= 0.01 and float(utarget[i]) >= 0.001):
       # if (float(redshift[i]) == 0): 
           # if(float(ntarget[i]) == 300):
               # if (float(ztarget[i]) == 0.1):
           # if (float(nhtarget[i]) == 21): #**
                      
                    #nhtarget[i] == 21 and rtarget[i] == 21.291
                        trainingResult2.append(training[i])
                        agntarget2.append(agntarget[i])
                        ztarget2.append(ztarget[i])
                        utarget2.append(utarget[i])
                        nhtarget2.append(nhtarget[i])
                        ntarget2.append(ntarget[i])
                        rtarget2.append(rtarget[i])
                    #    redshift2.append(redshift[i])


#transposed 
training2 = matrixTranspose(trainingResult2)


#{'CL4_20.3197m', 'FE4_2.86447m', 'FE5_20.8407m', 'FE4_2.83562m', 'AR3_21.8253m', 'CL4_11.7629m', 'FE5_25.9131m', 'AR5_7.89971m'}
 #fe and ar5
 #cl and ar5

#a = 'FE6_14.7670m'
#b =  'FE6_19.5527m'
#c = 'AR6_4.52800m'
#d = 'S8_9914.00A'


#a ='FE6_12.3074m'
#b ='FE6_19.5527m'
#c ='FE5_25.9131m'
#d ='FE6_1.34929m'
 
#a ='FE5_25.9131m'
#b ='CL4_20.3197m'
#c = 'FE6_12.3074m'
#d ='FE6_19.5527m'

#c = 'FE4_3.21932m'
#d = 'FE4_3.39109m'
# {'AR2_6.98337m', 'AR5_13.0985m', 'MG5_5.60700m', 'MG4_4.48712m', 'NE6_7.64318m'}
c = 'H1_2.16551m'
b = 'NE6_7.64318m'
a =  'S4_10.5076m'
d = 'AR6_4.52800m'
# 'S4_10.5076m', 'NE6_7.64318m'
#{'H1_2.16551m', 'FE2_5.33881m', 'AR2_6.98337m', 'AR5_13.0985m', 'MG5_5.60700m', 'MG4_4.48712m', 'NE6_7.64318m'}


xname = 'Log(' + a + ' / ' + b + ')'
yname = 'Log(' + c + ' / ' + d + ')'

xVal = []
yVal = []
for i in range(len(trainingResult2)):
    aVal = float(trainingResult2[i][names2.index(a)])
    bVal = float(trainingResult2[i][names2.index(b)])
    cVal = float(trainingResult2[i][names2.index(c)])
    dVal = float(trainingResult2[i][names2.index(d)])
    if (bVal != 0):
        xVal.append(aVal/bVal)
    else:
        xVal.append(0)
    
    if (dVal != 0):
        yVal.append(cVal/dVal)
    else:
        yVal.append(0)
        
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(agntarget2), toFloat(ztarget2), xname, yname,'AGN')
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(agntarget2), toFloat(ztarget2), xname, yname,'Z')
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(nhtarget2), toFloat(ztarget2), xname, yname,'NH')
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(rtarget2), toFloat(ztarget2), xname, yname,'R')
paperPlots(logFunction(xVal), logFunction(yVal), toFloat(utarget2), toFloat(ztarget2), xname, yname,'U')


#paperPlots(x,x2,y,y2,z,xname,xname2,yname,yname2,zname):
#{'K6_5.573m', 'K6_8.821m', 'SI7_2.481m', 'NA6_14.40m'}
#{'SI7_2.481m', 'K6_5.573m', 'MG7_9.033m', 'NA6_14.40m', 'FE7_2.629m'}

#create a plot Log(a/b) vs Log (c/d)
#plot all the lines and divide them by colors (starforming and non star forming)
