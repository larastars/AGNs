#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:52:39 2019

@author: larakamal
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


flights = sns.load_dataset("flights")
print(flights)
flights = flights.pivot("month", "year", "passengers")
a = flights[:1]
b = flights[1:2]
c = flights[2:3]

#ax = sns.heatmap(flights)
