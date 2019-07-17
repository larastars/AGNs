#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:20:34 2019

@author: larakamal
"""
import csv 
import os
#code that combines:
#.grd files (R, NH)
#.lst files (emission lines)
#.out files (U, the number after the string:'U(1.0----))
#reads the Agn, Z, n from directory 

################create a combination of n, Z, and AGN
n = [1000,300]
Z = [0.1,1]
agn = [0,5,10,20,40,60,80,100]
num = [1,2]

matrix = []
matrix.append(['num', 'n', 'Z', 'agn'])
for i in num:
    for j in n:
        for k in Z:
            for l in agn:
                matrix.append([i,j,k,l])


################extract R and NH from the .grd file 
#put the .grd files together
#extract num,Z,n,agn
#make a list of radius and stop colum nh 
for file in os.listdir("/confiles"):
    if file.endswith(".grd"):
        print(os.path.join("/mydir", file))

            
"""
# creates a new csv files 
with open('finalfileme.csv', mode='w') as file:
    outputwriter = csv.writer(file, delimiter=',')
    for i in range(len(matrix)):
        outputwriter.writerow(matrix[i])

"""