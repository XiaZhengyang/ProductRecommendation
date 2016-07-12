# coding=utf-8
import simplejson
import numpy as np
import scipy 
from numpy import array
from sklearn.cluster import KMeans


with open('../newData/new-data/Customer_info.json') as data_file:
	data = simplejson.load(data_file)

numberOfClusters = 4
alpha = 0.03
numberOfFeatures = 7
infoMatrix = np.zeros((1,numberOfFeatures),)
validSampleCount = 0


for i in range (0,108):
	if 'customerBaseExt' in data["clients"][i]:
		validSampleCount = validSampleCount+1
		infoMatrix.resize((validSampleCount,numberOfFeatures))
		
		#Age
		birthDay= data["clients"][i]["customerBaseExt"]["birthday"]
		birthYear = int(birthDay[0:4])
		infoMatrix[validSampleCount-1,0] = (2016- birthYear)/float(10)


		
		#Gender
		gender = data["clients"][i]["customerBaseExt"]["gender"]
		if gender == "Male":
			infoMatrix[validSampleCount-1,1] = 1
		if gender == "Female":
			infoMatrix[validSampleCount-1,1] = 0 
		

		#Maritial Status
		if data["clients"][i]["customerMarriage"]["isMarried"] == "Married":
			infoMatrix[validSampleCount-1,2] =1
		else:
			infoMatrix[validSampleCount-1,2] = 0
		educationString = data["clients"][i]["customerEducations"][0]["education"]


		#Education Background
		if educationString == "Bachelor":
			infoMatrix[validSampleCount-1,3] = 4
		elif educationString =="College":
			infoMatrix[validSampleCount-1,3] = 2
		elif educationString == "Senior":
			infoMatrix[validSampleCount-1,3] = 1
		else:
			infoMatrix[validSampleCount-1,3] = 0

		#Monthly Income & no. of dependent(s)
		if 'customerAssetsIncome' in data["clients"][i]:
			monthlyNetIncome = data["clients"][i]["customerAssetsIncome"]["monthlySalary"] + data["clients"][i]["customerAssetsIncome"]["monthlyIncomeOther"] - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]
			annualNetIncome = data["clients"][i]["customerAssetsIncome"]["yearlyIncome"] - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]*12
			infoMatrix[validSampleCount-1,4] = max(monthlyNetIncome,annualNetIncome/12)/float(10000)
			infoMatrix[validSampleCount-1,5] = data["clients"][i]["customerAssetsIncome"]["dependentNumber"]

		#Vehicle Ownership
		if data["clients"][i]["customerAssetsVehicles"]:
			infoMatrix[validSampleCount-1,6] = float(data["clients"][i]["customerAssetsVehicles"][0]["vehiclePurchasePrice"])/float(100000)



print (infoMatrix)


theta = np.zeros((numberOfFeatures,1),)
for i in range (0,numberOfFeatures):
	theta[i]=1			#initialize vector theta



def costFunction(originalMatrix,thetaInput):
	adjustedMatrix = (originalMatrix)*(np.transpose(thetaInput))
	kmeansObject = KMeans(numberOfClusters,n_init=50)
	kmeansObject.fit(adjustedMatrix)
	return kmeansObject.inertia_

def partialDerivative(infoMatrix, theta, i,h=0.005):
	newThetaBig = np.zeros((numberOfFeatures,1),)
	newThetaSmall = np.zeros((numberOfFeatures,1),)
	for j in range(0,numberOfFeatures):
		newThetaBig[j] = theta[j]
		newThetaSmall[j] = theta[j]
	newThetaSmall[i] = theta[i]-h
	newThetaBig[i] = theta[i] +h			
	return (costFunction(infoMatrix,newThetaBig)-costFunction(infoMatrix,newThetaSmall))/(2*h)
	

def updateTheta(data, thetaInput):
	tempStoreNewTheta = np.zeros((numberOfFeatures,1),)
	for i in range(0,numberOfFeatures):
		tempStoreNewTheta[i] = thetaInput[i] - alpha*(partialDerivative(data,thetaInput,i))
	return tempStoreNewTheta









'''



weightedMatrix = infoMatrix*np.transpose(theta)

newClient = np.zeros(numberOfFeatures)
for i in range(0,numberOfFeatures):
	newClient[i] = raw_input("Please enter the feature of newClient-->")

print newClient
print "The new client belongs to cluster number ", np.argmin(distance)

'''