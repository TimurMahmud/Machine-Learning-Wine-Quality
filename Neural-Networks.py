# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:43:26 2022

@author: timur
"""

# import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# read in data and store in Pandas DataFrame
df = pd.read_csv("winequality-red.csv")

# separate data into input features and target labels
X = df.drop("quality",axis=1)
y = df["quality"]

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=7)

# standardize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create instance of MLPClassifier and specify model hyperparameters
clf = MLPClassifier(max_iter=5000,solver="lbfgs", alpha=1e-4, hidden_layer_sizes=(3,3), random_state=0)

# fit model to training data
clf.fit(X_train, y_train)

# use model to make predictions on test set
preds = clf.predict(X_test)

# print classification report and accuracy score
print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds))
