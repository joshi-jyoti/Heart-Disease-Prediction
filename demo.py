import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the CSV data
heart_data = pd.read_csv("heart.csv")  # Ensure the file is in the same directory as your script

# print first five row
print(heart_data.head())

# print the last five rows
print(heart_data.tail())

# to check the total nnumber of rows and column
print(heart_data.shape)

# details of  the data 
print(heart_data.info())

# Generating statistical insights from the data
print(heart_data.describe())

#checking the total number of target as 0 and 1
print(heart_data['target'].value_counts())

# 1 represent defective heart and 0 represent healthy heart

# splitting the features and target 
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# printing the value of X which contain 13 rows
print(X)

# printing the value of Y which contain target
print(Y)

# splitting the data into Training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Model Training

# Logistic Regression

model = LogisticRegression()

# training the logisticRegression model with Training data
model.fit(X_train, Y_train)

# Evaluation of Model

# Accuracy Score

# accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction ,Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on Test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction ,Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

# Building a Predictive System

input_data = (44,0,2,108,141,0,1,175,0,0.6,1,0,2)

# change the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we are predicting for one data

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The person is healthy, no heart disease')
else:
    print('The person has heart disease')
