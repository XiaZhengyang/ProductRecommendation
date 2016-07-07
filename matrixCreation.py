# coding=utf-8
import simplejson
import numpy as np
import scipy 
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from numpy import array
np.seterr(divide='ignore', invalid='ignore')


with open('../clientInformation.json') as data_file:
	data = simplejson.load(data_file)

alpha = 0.000001
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
	whitenedMatrix[i,1] = int(1);
	whitenedMatrix[i,2] = int(1);

print (whitenedMatrix)
numberOfClusters = 2
resultOfKmeans1 = kmeans(whitenedMatrix,numberOfClusters)
clusteringResult =  kmeans2(whitenedMatrix,resultOfKmeans1[0],minit='points')

print ('===The k-means result is===')
print (clusteringResult)


theta = np.zeros((numberOfFeatures,1),)
for i in range (0,numberOfFeatures):
	theta[i]=1			#initialize vector theta



def costFunction(originalMatrix,thetaInput):
	adjustedMatrix = (originalMatrix)*(np.transpose(thetaInput))
	return kmeans(adjustedMatrix,numberOfClusters)[1]

def deriv(infoMatrix, theta, h=0.005):
	partialDerivativeArray = np.zeros(7)
	for i in range(7):
		newThetaBig = np.zeros((numberOfFeatures,1),)
		for j in range(0,numberOfFeatures):
			newThetaBig[j] = theta[j]
		newThetaBig[i] = theta[i] +h
		newThetaSmall = np.zeros((numberOfFeatures,1),)
		for j in range(0,numberOfFeatures):
			newThetaSmall[j] = theta[j]
		newThetaSmall[i] = theta[i]-h
		
		partialDerivativeArray[i] = (costFunction(infoMatrix,newThetaBig)-costFunction(infoMatrix,newThetaSmall))/(2*h)
	return partialDerivativeArray

def updateTheta(data, theta):
	#print theta
	tempStoreNewTheta = np.zeros((numberOfFeatures,1),)
	for i in range(0,numberOfFeatures):
		tempStoreNewTheta[i] = theta[i] - alpha*(deriv(data,theta)[i])
	for i in range(0, numberOfFeatures):
		theta[i] = tempStoreNewTheta[i]
	#print theta
	return costFunction(data, theta)
	#print "====="
'''
cost = np.zeros(1000)
for i in range(1000):

	cost[i] = updateTheta(infoMatrix,theta);
	print cost[i]
	if ((i>1) & (cost[i-1] - cost[i] < 0.001)):
		print cost[i-1], '   ', cost[i]
		break
print theta
'''

print minimize(costFunction(infoMatrix,theta),theta);




