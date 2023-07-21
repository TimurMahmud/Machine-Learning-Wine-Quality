# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:00:32 2022

@author: timur
"""

# Import necessary libraries and modules
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the dataset from a CSV file using pandas
df = pd.read_csv("winequality-red.csv")

# Separate the features and target variable
X = df.drop("quality", axis=1)  # features
y = df["quality"]  # target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# Scale the training and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a logistic regression model and train it on the training data
lr = LogisticRegression(max_iter=1000, penalty="l2")
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing data
preds = lr.predict(X_test)

# Print the classification report and accuracy of the predictions
print(classification_report(y_test, preds))
print(accuracy_score(y_test, preds))
