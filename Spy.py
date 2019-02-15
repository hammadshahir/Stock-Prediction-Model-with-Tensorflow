# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 23:46:56 2019

@author: Hammad
"""

import pandas as pd
import numpy as np
import tensorflow as tf

spyTrainData = 'Resources/CSV_Files/SPYTrainData.csv'
spyTestData = 'Resources/CSV_Files/SPYTestData.csv'

currentTrainData = spyTrainData
currentTestData = spyTestData

trainDataPoints = 1259
testDataPoints = 62

learningRate = 0.1
num_epochs = 100

# Creating load file function
def loadStockData(stockName, numDataPoints):
    data = pd.read_csv(stockName, 
                       skiprows = 0,
                       nrows = numDataPoints,
                       usecols = ['Close', 'Open', 'Volume']
                       )

    finalPrice = data['Close'].astype(str).str.replace(',','').astype(np.float) 
    openPrice = data['Open'].astype(str).str.replace(',','').astype(np.float) 
    volume = data['Volume']
    return finalPrice, openPrice, volume
    
#Calculating Closing and Opening Price Difference

def calculatePriceDifference(finalPrice, openPrice):
    priceDifferences = []
    for dif in range(len(finalPrice) - 1):
        priceDifference = openPrice[dif + 1] - finalPrice[dif]
        priceDifferences.append(priceDifference)
    return priceDifferences

#Calculatin loss function (Calculating Accuracy)
    
def calculateAccuracy(expectedValues, actualValues):
    numCorrectGuess = 0
    for accurate in range(len(actualValues)):
        if actualValues[accurate] < 0 < expectedValues[accurate]:
            numCorrectGuess += 1
        elif actualValues[accurate] > 0 > expectedValues[accurate]:    
            numCorrectGuess += 1
    return (numCorrectGuess/len(actualValues)) * 100
            
# Training data set

trainFinalPrices, trainOpeningPrices, trainVolumes = loadStockData(currentTrainData, trainDataPoints)
trainPriceDifferences =  calculatePriceDifference(trainFinalPrices, trainOpeningPrices)
trainVolumes = trainVolumes[:-1]

# Testing data set
testFinalPrices, testOpeningPrices, testVolumes = loadStockData(currentTestData, testDataPoints)
testPriceDifferences = calculatePriceDifference(testFinalPrices, testOpeningPrices)
testVolumes = testVolumes[:-1]

# Regression y = Wx + b

x = tf.placeholder(tf.float32, name='x')
W = tf.Variable([.1], name='W')
b = tf.Variable([.1], name='b') 

y = (W * x) + b
y_predicted = tf.placeholder(tf.float32, name='y_predict')

# Loss Optimizer
loss = tf.reduce_sum(tf.square(y - y_predicted))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

# Run Sessions

session = tf.Session()
session.run(tf.global_variables_initializer())

# Running epochs
for num in range(num_epochs):
    session.run(optimizer, feed_dict = {x:trainVolumes, y_predicted:trainPriceDifferences})   

# Measuring Results
results = session.run(y, feed_dict={x:testVolumes})
accuracy = calculateAccuracy(testPriceDifferences, results)
print("Model Accuracy: {0:.2f}%".format(accuracy))
