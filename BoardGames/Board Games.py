# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 23:43:40 2021

@author: Akash Dwivedi
"""
import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#Load the data
games = pandas.read_csv("games.csv")

#print the names of the columns in games
print(games.columns)
print(games.shape)

#MAke a histogram of all the ratings in the average raiting column
plt.hist(games["average_rating"])
plt.show()

#but we can see in histogram that 25k+ games has zero rating which is not digestable
#hence explore more

#Print the first row of all the games with zero scores
print(games[games["average_rating"]==0].iloc[0])

#Print the first row of games with score greater than zero
print(games[games["average_rating"]>0].iloc[0])

#Remove any rows without user reviews
games= games[games["users_rated"]>0]

#Remove any rows with  missing values
games = games.dropna(axis=0)

#Make a histogram of all the average ratings
plt.hist(games["average_rating"])
plt.show()

#Now its better than previous histogram

#Correlation Matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()
#This coorelation fugure will tell us how features are corealted, white means highly correlated, black means no correlation
#this correlation will help in generating ML algorithm

#Now preProcessing comes
#get all the columns from dataframe
columns = games.columns.tolist()

#Filter the columns the data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#Store the variable we will be predicting on
target = "average_rating"

#Generate training and test datasets
from sklearn.model_selection import train_test_split

#Genrate Training set
train = games.sample(frac=0.8, random_state =1)   #80% data for training

#Select anything not in training set and put it in test set
test= games.loc[~games.index.isin(train.index)]

#print shapes
print(train.shape)
print(test.shape)
#we have 45k+ games in training set and 11k+ games in test data


### Machine Learning Algo comes now

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initializa the model class
LR = LinearRegression()

#Fit the model  the training data
LR.fit(train[columns], train[target])

#Generate predictions for the test set
predictions = LR.predict(test[columns])

#Compute Error between our test predicitons and actual values
print(mean_squared_error(predictions, test[target]))
#Error is 2.0788 ;large so not good 
#We will try to reduce it using nonlinear model
# hence linear regression does not work! and it makes sense also as features are not correlated

# Import the random forest model (Nonlinear)
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf =10, random_state =1)

#Fit to the data
RFR.fit(train[columns], train[target])



#Make predicitons

predictions = RFR.predict(test[columns])

#Compute the error btween the test prediciotns and actualvalues
print(mean_squared_error(predictions, test[target]))


## Yeah!! the error now is 1.4458 , means after random forest regressor. the error got reduced by half


test[columns].iloc[0]

#Make prediction with both the models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

#print  out the prediciton
print(rating_LR)
print(rating_RFR)


print(test[target].iloc[0])

### Completed
