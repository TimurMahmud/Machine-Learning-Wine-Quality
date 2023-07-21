# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:36:23 2022

@author: mitchmas
"""

#load necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#load the dataset
df = pd.read_csv("winequality-red.csv")

#Seperate quality column for target labels, every other column as input features
X = df.drop("quality",axis=1)
y = df["quality"]

#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=7) 

#set up gaussian naive bayes as data in input is continuous
gnb_model = GaussianNB()

#fit and test GNB
gnb_model.fit(X_train, y_train)

#create model for predictions on test set
y_pred_gnb = gnb_model.predict(X_test)

#display classification report and accuracy score
print(classification_report(y_test, y_pred_gnb))
print('GNB Accuracy: %.3f' % accuracy_score(y_test, y_pred_gnb))
