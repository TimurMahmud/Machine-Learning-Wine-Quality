# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:58:58 2022

@author: timur
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# load winequality dataset into dataframe
wine = pd.read_csv("winequality-red.csv")


# Use all columns as data features exept quality column which is the target
X = wine[ wine.columns[wine.columns!='quality'] ] 
# Select quality as target
y = wine['quality']

# Segment as testing and training 
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 



# kNN setup
knn_model = KNeighborsClassifier(n_neighbors=1)

# kNN test
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))


#The following models have undergone experimentation to determine the appropriate parameter values. 
#The specific values are not provided, but you can try varying them and plot the resulting curves.

# training
knn_model.fit(X_train,y_train)

# predict using testing data
y_pred = knn_model.predict(X_test)


# Bar chart predicted values vs expected to show the error
def plt_error(pred, y):
    fig, ax = plt.subplots()
    
    # Bar chart vs actual values    
    ax.bar(pred, y, alpha=0.7, color="tab:blue", label="expected vs prediction")

    fig.legend()
    plt.show()
    
plt_error(y_pred, y_test)

