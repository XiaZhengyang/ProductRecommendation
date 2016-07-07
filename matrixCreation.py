# coding=utf-8
import simplejson
import numpy as np
import scipy 
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import kmeans2
from numpy import array
np.seterr(divide='ignore', invalid='ignore')


with open('../clientInformation.json') as data_file:
	data = simplejson.load(data_file)
print (data)



numberOfFeatures = 7
infoMatrix = np.zeros((4,numberOfFeatures),int)


for i in range (0,4):
	birthDay= data["clients"][i]["customerBaseExt"]["birthday"]
	birthYear = int(birthDay[0:4])
	infoMatrix[i,0] = 2016- birthYear
	#infoMatrix[i,1] =int(data["clients"][i]["customerBaseExt"]["householdRegisterAddrCity"])
	if data["clients"][i]["customerMarriage"]["isMarried"] == "Married":
		infoMatrix[i,2] =1
	else:
		infoMatrix[i,2] = 0
	educationString = data["clients"][i]["customerEducations"][0]["education"]

	if educationString == "Bachelor":
		infoMatrix[i,3] = 4
	elif educationString =="College":
		infoMatrix[i,3] = 3
	elif educationString == "Senior":
		infoMatrix[i,3] = 2
	else:
		infoMatrix[i,3] = 1

	assetIncome = data["clients"][i]["customerAssetsIncome"]["monthlySalary"] + data["clients"][i]["customerAssetsIncome"]["monthlyIncomeOther"] - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]
	infoMatrix[i,4] = assetIncome;

	infoMatrix[i,5] = data["clients"][i]["customerAssetsIncome"]["dependentNumber"]

	if data["clients"][i]["customerAssetsVehicles"]:
		infoMatrix[i,6] = data["clients"][i]["customerAssetsVehicles"][0]["vehiclePurchasePrice"]


print (infoMatrix)

whitenedMatrix = whiten(infoMatrix)
for i in range (0,4):
	whitenedMatrix[i,1] = int(0);
	whitenedMatrix[i,2] = int(1);

print (whitenedMatrix)
numberOfClusters = 2
resultOfKmeans1 = kmeans(whitenedMatrix,numberOfClusters)
clusteringResult =  kmeans2(whitenedMatrix,resultOfKmeans1[0],minit='points')

print ('===The k-means result is===')
print (clusteringResult)


theta = np.zeros((numberOfFeatures,1),)


for i in range (0,numberOfFeatures):
	theta[i]=1/float(numberOfFeatures)			#initialize vector theta
print (theta)

adjustedMatrix = infoMatrix


print (adjustedMatrix)
	


def costFunction(infoMatrix,theta):
	return kmeans(infoMatrix,numberOfClusters)[1]


maximumIteration = 30




print ("The current clustering result is:" )
print (kmeans2(whitenedMatrix,kmeans(whitenedMatrix,numberOfClusters)[0],minit='points')[1])
print ("With the cost function value being " )
print (costFunction(infoMatrix,theta))







