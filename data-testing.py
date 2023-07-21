# -*- coding: utf-8 -*-
"""
Created on Sun Dec 4 17:31:26 2022

@author: timur
"""

import pandas as pd

df = pd.read_csv("winequality-red.csv")

# Shows the features and target with some data
print(df.head())

# Shows correlation between features and target
print(df.corr()['quality'].sort_values(ascending=False)[1:])
      
# Shows columns
print(df.columns)

# Checks for null data inputs
print(df.isnull().sum().sum())

# Shows Non-null count and data types of each column
print(df.info())

# Transpose index and columns
print(df.describe().T)