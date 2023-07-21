#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pav2001
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score



#Load dataset
wine = pd.read_csv("winequality-red.csv")

#Train test slit
X = wine.drop('quality',axis=1)
X= StandardScaler().fit_transform(X)
y= wine[['quality']]

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

X_train.shape

plt.hist(wine["quality"])
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(wine.corr(), annot=True, cmap="PuOr")
plt.show()


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create a Decision Tree
dtree_basic = DecisionTreeClassifier(max_depth=10)
# Fit the training data
dtree_basic.fit(X_train,y_train)
# Predict based on test data
y_preds = dtree_basic.predict(x_test)


# Calculate Accuracy
accuracy_value = metrics.accuracy_score(y_test,y_preds)
accuracy_value
# Create and print confusion matrix
confusion_matrix(y_test,y_preds)
print(classification_report(y_test,y_preds))


score = accuracy_score(y_preds, y_test)
print("Accuracy", score)


# Calculate the number of nodes in the tree
dtree_basic.tree_.node_count

# Create a Parameter grid

param_grid = {
    'max_depth' : range(4,20,4),
    'min_samples_leaf' : range(20,200,40),
    'min_samples_split' : range(20,200,40),
    'criterion' : ['gini','entropy'] 
}
n_folds = 5

dtree = DecisionTreeClassifier()
grid = GridSearchCV(dtree, param_grid, cv = n_folds, n_jobs = -1,return_train_score=True)
grid.fit(X_train,y_train)

cv_result = pd.DataFrame(grid.cv_results_)
cv_result.head()
grid.best_params_
grid.best_score_

best_grid = grid.best_estimator_
best_grid
best_grid.fit(X_train,y_train)

y_preds = best_grid.predict(x_test)

# Calculate Accuracy
accuracy_value = metrics.accuracy_score(y_test,y_preds)
accuracy_value
# Create and print confusion matrix
confusion_matrix(y_test,y_preds)

print(classification_report(y_test,y_preds))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_preds, y_test)
print("Accuracy", score)

