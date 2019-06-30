#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:03:54 2019

@author: larakamal
"""
import csv 

def duplicate(items): 
    unique = [] 
    dup = []
    for item in items: 
        if item not in unique: 
            unique.append(item) 
        else:
            dup.append(item)
    return unique, dup


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
agn1 = []
R1 = []
N1 = []
U1 = []
Z1 = []
n1 = []
rest1 = []

Z2 = []
U2 = []
NH = []
n2 =[]
R2 = []
agn2 = []
rest2 = []

file_reader = open('finalfile2.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[0] != ''):
        agn1.append(row[1:2])
        R1.append(row[2:3])
        N1.append(row[3:4])
        U1.append(row[4:5])
        Z1.append(row[5:6])
        n1.append(row[6:7])
        rest1.append(row[7:])
        

file_reader = open('grid_colors_local.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[0] != ''):
        Z2.append(row[0:1])
        U2.append(row[1:2])
        R2.append(row[2:3])
        n2.append(row[3:4])
        NH.append(row[4:5])
        agn2.append(row[5:6])
        rest2.append(row[6:])
        

names = ['Z', 'U', 'N', 'n', 'R', 'agn']
names = names + rest2[0] + rest1[0]
matrix = []
matrix.append(names)

#remove label row 
agn1 = agn1[1:] 
R1 = R1[1:] 
N1 = N1[1:] 
U1 = U1[1:] 
Z1 = Z1[1:] 
n1 = n1[1:] 
rest1 = rest1[1:] 

Z2 = Z2[1:] 
U2 = U2[1:] 
NH = NH[1:] 
n2 = n2[1:] 
R2 = R2[1:] 
agn2 = agn2[1:] 
rest2 = rest2[1:] 

test = []
for i in range(len(Z2)):
    curr= [Z2[i][0],U2[i][0],NH[i][0],n2[i][0],R2[i][0],agn2[i][0]]
    test.append(curr)

act, d = duplicate(test)
for i in d:
    print(i)
    
test2 = []
for i in range(len(Z1)):
    curr= [Z1[i][0],U1[i][0],N1[i][0],n1[i][0],R1[i][0],agn1[i][0]]
    test.append(curr)

act2, d2 = duplicate(test2)
for i in d2:
    print(i)
    
print('done')
"""

[N11] = matrixTranspose(R1)
[NH1] = matrixTranspose(R2)

N111 = set(N11)
NH11 = set(NH1)
#print(N111)
#print("")
#print(NH11)
a = set(N11).intersection(set(NH1))
print(a)
print(len(Z1))
print(len(R1))
print(len(N1))
print(len(U1))
print(len(n1))
print(len(agn1))
print(len(rest1))

print(len(U2))
print(len(Z2))
print(len(NH))
print(len(n2))
print(len(agn2))
print(len(rest2))


"""


num = []
#print(Z1[4][0])
rounding = 7
for i in range(len(Z1)):
    for j in range(len(Z2)):
        if (float(Z1[i][0]) == float(Z2[j][0])):
           # print('1')
            if (float(U1[i][0]) == float(U2[j][0])):
              #  print('2')
                if(float(agn1[i][0]) == float(agn2[j][0])):
                 #   print('3')
                    if(float(n1[i][0]) == float(n2[j][0])):
                        #print('4')
                        if(float(N1[i][0]) == float(NH[j][0])):
                         #   print('5')
                            if(float(R1[i][0]) == float(R2[j][0])):
                                num.append(j)
                                #print('6')
                                curr = []
                                curr.append(Z2[j][0])
                                curr.append(U2[j][0])
                                curr.append(NH[j][0])
                                curr.append(n2[j][0])
                                curr.append(R2[j][0])
                                curr.append(agn2[j][0])
                                curr = curr + rest2[j] + rest1[i]
                                if (curr not in matrix):
                                    matrix.append(curr)
      

print(len(R1))
print(len(R2))
print(len(matrix))
print(len(num))
num2, dup = duplicate(num)
print(len(dup))
print(dup)


with open('colorgridoutput1.csv', mode='w') as file:
    outputwriter = csv.writer(file, delimiter=',')
    for i in range(len(matrix)):
        outputwriter.writerow(matrix[i])
