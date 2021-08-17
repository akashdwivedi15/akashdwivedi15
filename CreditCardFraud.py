# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:59:36 2021

@author: Akash Dwivedi
"""

import sys
import pandas
import matplotlib
import seaborn
import sklearn
import scipy
import numpy

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('seaborn: {}'.format(sklearn.__version__))

#import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset of credit card
data = pd.read_csv('creditcard.csv')

#Explore the Data
print(data.columns)
print(data.shape)

print(data.describe())

#Now since dataset is large so we will take some sample of it that is 10% of data we will take
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)   #now data which was 284807 earlier will become 28481

#Plot histogram of each feature
data.hist(figsize= (20,20))
plt.show()


#Determine number of Fraud Cases in dataset

Fraud = data[data['Class'] ==1]
Valid = data[data['Class'] ==0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases :{}'.format(len(Fraud)))
print('Valid Cases :{}'.format(len(Valid)))

#Fraud Cases = 28432 and Valid Cases = 49

#Correlation Matrix to check the relationship
corrmat = data.corr()
fig = plt.figure(figsize =  (12,9))
sns.heatmap(corrmat, vmax=.8, square = True)
plt.show()

#Get All the columns from the dataframe
columns = data.columns.tolist()

#Filter the columns to remove data we dont want
columns = [c for c in columns if c not in ["Class"]]

#Store the  variable we'll be predicting on
target = "Class"
X = data[columns]
Y = data[target]

#Print the shapes of X and Y
print(X.shape)
print(Y.shape)



######### Now implementation of ALogrithm for ML

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#Defining a random state
state = 1

#### Defining the Outlier Detections method
classifiers = {
    "Isolation Forest" : IsolationForest(max_samples= len(X), contamination = outlier_fraction, random_state= state), 
    "Local Outlier Factor" :LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
                                               }

# Fit the Model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor" : 
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
        
        
#Reshape the prediciton values to 0 for valid, 1 for fraud

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

n_errors = (y_pred != Y).sum()


#Run cLassification metrics
print('{}: {}'.format(clf_name, n_errors))

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))

#But we have false postive as 0.02 (precision) for  fraud cases and false negative (recall)
 
      