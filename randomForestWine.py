import pandas as pd
import numpy as np
import os
from pprint import pprint

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

#set your path
path = "."

#read in the data as csv
filename_read = os.path.join(path, "winequality-red.csv")
dataset = pd.read_csv(filename_read)

#print(dataset.shape)
#print(dataset[:5])

dataset.columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
             'density','pH','sulphates','alcohol','target']

#Encode the feature values which are strings to integers
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

# Create our X and y data    
result = []
for x in dataset.columns:
    if x != 'target':
        result.append(x)

X = dataset[result].values
y = dataset['target'].values

#print(X[:5])

#Instantiate the model with 10 trees and entropy as splitting criteria
Random_Forest_model = RandomForestClassifier(n_estimators=10,criterion="entropy")

#Training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

#Train the model
Random_Forest_model.fit(X_train, y_train)

#make predictions
y_pred = Random_Forest_model.predict(X_test)

#print(y_pred[:5])
#print(y_test[:5])

#Calculate accuracy metric
accuracy = accuracy_score(y_pred, y_test)
print('The accuracy is: ',accuracy*100,'%')

digits = load_digits()

print(digits.data.shape)
print(digits.images.shape)

X = digits.data
y = digits.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=7)

#builds a rf classifier over 128 estimators
#rf_model = RandomForestClassifier(n_estimators=128,criterion="entropy")
#rf_model.fit(Xtrain, ytrain)
#y_model = rf_model.predict(Xtest)

#accuracy = accuracy_score(ytest, y_model)

style.use('fivethirtyeight')

fig = plt.figure(figsize=(15,10))

#investigates the accuracy over a range of estimators plotting the result
#THIS TAKES A WHILE TO RUN!!
accuracy_data = []
nums = []
for i in range(1,128):
    rf_model = RandomForestClassifier(n_estimators=i,criterion="entropy")
    rf_model.fit(Xtrain, ytrain)
    y_model = rf_model.predict(Xtest)
    accuracy = accuracy_score(ytest, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)
    
print(accuracy_data)
plt.plot(nums,accuracy_data)
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.show()