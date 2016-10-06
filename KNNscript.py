# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import csv
import random
import math
import operator


titanicKNN = pd.read_csv("D:/Kaggle/Titanic/train.csv")

  
def intColumns(filename):
    csvdataframe = pd.read_csv(filename)
    intdata = pd.DataFrame()
    count=0
    for x in csvdataframe.columns:
        if csvdataframe[x].dtype == 'int64' or csvdataframe[x].dtype == 'float64':
            intdata.insert(count,x,csvdataframe[x])
            count+=1
    return(intdata)
    
data = pd.DataFrame()
count = 0    
for x in titanicKNN.columns:
    if x != 'Survived':
        data.insert(0,x,titanicKNN[x])
        count+=1
    if x== 'Survived':
        data.insert(count,x,titanicKNN[x])
        count+=1
titanicKNN = data

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingData=pd.read_csv("D:/Kaggle/Titanic/train.csv")
	testData=pd.read_csv("D:/Kaggle/Titanic/test.csv")
    	trainingSet=intColumns("D:/Kaggle/Titanic/train.csv")
      testSet=intColumns("D:/Kaggle/Titanic/test.csv")
      #Keep Survived in the last of training set
      data = pd.DataFrame()
      count = 0 
      for x in trainingSet.columns:
          if x != 'Survived':
              data.insert(0,x,trainingSet[x])
              count+=1
          if x== 'Survived':
              data.insert(count,x,trainingSet[x])
              count+=1
      trainSet = data
      trainingSet = trainSet.as_matrix()
      #Keep Survived in the last of test set
      data = pd.DataFrame()
      count = 0 
      for x in testSet.columns:
          if x != 'Survived':
              data.insert(0,x,testSet[x])
              count+=1
          if x== 'Survived':
              data.insert(count,x,testSet[x])
              count+=1
      testingSet = data
      testSet = testingSet.as_matrix()
      
	# generate predictions
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		testSet[x][-1] = result
	#	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	#accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')
        
main()