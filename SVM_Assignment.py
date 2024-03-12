# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:56:04 2024

@author: lapra
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 15)

# Load and check the data

# 1
path =  r"C:\Users\lapra\OneDrive\Desktop\SupLearning"
filename = 'breast_cancer.csv'
fullpath = os.path.join(path,filename)

data_nicholas = pd.read_csv(fullpath, sep = ',')

# Pre-process and visualize the data 

# 2a
print(data_nicholas.dtypes)

# 2b
missing_values = data_nicholas.isnull().sum()
for column, count in missing_values.items():
    print(f"{column}: {count} missing values")

# 2c
print(data_nicholas.describe().round(2))

# 3
data_nicholas['bare'].replace('?', np.nan, inplace = True)
data_nicholas['bare'] = data_nicholas['bare'].astype(float)

print(data_nicholas.dtypes)

# 4
data_nicholas.fillna(data_nicholas.median(), inplace = True)

# 5
data_nicholas = data_nicholas.drop(columns = ['ID'])

# 6 Creating a single plot with multiple plots
data_nicholas.hist(figsize = (9, 10), bins = 10)
plt.title("nicholas_Weight")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

from sklearn.model_selection import train_test_split

# 7
X = data_nicholas.drop('class', axis = 1)
y = data_nicholas['class']

# 8 301266745
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

# Build Classification Models - Support vector machine classifier with linear kernel

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 9
clf_linear_nicholas = SVC(kernel = 'linear', C = 0.1, random_state = 45)
clf_linear_nicholas.fit(X_train, y_train)

y_train_pred = clf_linear_nicholas.predict(X_train)
y_test_pred = clf_linear_nicholas.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# 10 + 11
print(f"Training Accuracy (linear kernel): {accuracy_train.round(4)}")
print(f"Testing Accuracy (linear kernel): {accuracy_test.round(4)}")

# 12 rbf kernel
clf_linear_nicholas = SVC(kernel = 'rbf', random_state = 45)
clf_linear_nicholas.fit(X_train, y_train)

y_train_pred = clf_linear_nicholas.predict(X_train)
y_test_pred = clf_linear_nicholas.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy (rbf kernel): {accuracy_train.round(4)}")
print(f"Testing Accuracy (rbf kernel): {accuracy_test.round(4)}")

# 13 poly kernel
clf_linear_nicholas = SVC(kernel = 'poly', random_state = 45)
clf_linear_nicholas.fit(X_train, y_train)

y_train_pred = clf_linear_nicholas.predict(X_train)
y_test_pred = clf_linear_nicholas.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy (poly kernel): {accuracy_train.round(4)}")
print(f"Testing Accuracy (poly kernel): {accuracy_test.round(4)}")

# 14 sigmoid kernel
clf_linear_nicholas = SVC(kernel = 'sigmoid', random_state = 45)
clf_linear_nicholas.fit(X_train, y_train)

y_train_pred = clf_linear_nicholas.predict(X_train)
y_test_pred = clf_linear_nicholas.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy (sigmoid kernel): {accuracy_train.round(4)}")
print(f"Testing Accuracy (sigmoid kernel): {accuracy_test.round(4)}")

# Part 2 of assignment
# 1
data_nicholas_df2 = pd.read_csv(fullpath, sep = ',')

# 2
data_nicholas_df2['bare'].replace('?', np.nan, inplace = True)
data_nicholas_df2['bare'] = data_nicholas_df2['bare'].astype(float)

# 3
data_nicholas_df2 = data_nicholas_df2.drop(columns = ['ID'])

# 4
X = data_nicholas_df2.drop('class', axis = 1)
y = data_nicholas_df2['class']

# 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 6 a + b
imputer = SimpleImputer(strategy = 'median')

scaler = StandardScaler()

# 7
num_pipe_nicholas = Pipeline(
    [
     ('imputer', imputer),
     ('scaler', scaler)
    ])

# 8
pipe_svm_nicholas = Pipeline(
    [
     ('preprosseing', num_pipe_nicholas),
     ('svm', SVC(random_state = 45))
     ])

# 10
param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svm__degree': [2, 3]
}

from sklearn.model_selection import GridSearchCV

# 12
grid_search_nicholas = GridSearchCV(estimator = pipe_svm_nicholas, param_grid = param_grid, scoring = 'accuracy', refit = True, verbose = 3)

# 14
grid_search_nicholas.fit(X_train, y_train)

# 15 + 16
print(f"Best Parameters: {grid_search_nicholas.best_params_}")
print(f"Best Estimator: {grid_search_nicholas.best_estimator_}")

# 17
best_model_nicholas = grid_search_nicholas.best_estimator_

# 18
best_model_nicholas.fit(X_train, y_train)
accuracy = best_model_nicholas.score(X_test, y_test)
print(f"Accuracy of Best Model Configuration: {accuracy.round(4)}")

from joblib import dump

# 19 + 20
dump(best_model_nicholas, 'best_model_nicholas.joblib')
dump(pipe_svm_nicholas, 'pipeline_nicholas.joblib')








