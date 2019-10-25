# Predicting which way a scale is tipped or it is balanced

#Import libraries 
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

#Importing datset
dataset = pd.read_csv('balance_scale.csv')
x = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, 4].values

#Categorical dats y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Creating raining and test sets
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#Fitting the svm regression model to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix - tp predict accuracy of model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
