# coding=utf-8
import simplejson
import numpy as np
import scipy 
from numpy import array
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.cluster.vq import whiten


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
		infoMatrix[validSampleCount-1,0] = (2016- birthYear)


		
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
			infoMatrix[validSampleCount-1,4] = max(monthlyNetIncome,annualNetIncome/12)/float(1000)
			infoMatrix[validSampleCount-1,5] = data["clients"][i]["customerAssetsIncome"]["dependentNumber"]

		#Vehicle Ownership
		if data["clients"][i]["customerAssetsVehicles"]:
			infoMatrix[validSampleCount-1,6] = float(data["clients"][i]["customerAssetsVehicles"][0]["vehiclePurchasePrice"])/float(1000)


#Feature Scaling and record the scaling coefficients
print (infoMatrix)
scaledMatrix = whiten(infoMatrix)
scalingCoefficient = np.zeros((numberOfFeatures,1),)
for i in range(numberOfFeatures):
	j =0 			#traverse all training examples until a non-zero number if found
	while (infoMatrix[j,i]==0):
		j=j+1
	scalingCoefficient[i] = scaledMatrix[j,i]/infoMatrix[j,i]	
print scalingCoefficient
print scaledMatrix


#Perform optimization
kmObject = KMeans(4,n_init=75)
kmObject.fit(scaledMatrix)
print kmObject.labels_, kmObject.inertia_

#Input a new client information and scale it
newClient = np.zeros(numberOfFeatures)
for i in range(0,numberOfFeatures):
	newClient[i] = raw_input("Please enter the feature of newClient-->")
newClientScaled = np.zeros(numberOfFeatures)
for i in range(numberOfFeatures):
	newClientScaled[i] = newClient[i]*scalingCoefficient[i]
print newClientScaled

#Calculate the distances of new client to the cluster-centroids
distance = np.zeros(numberOfClusters)
for i in range(numberOfClusters):
	for j in range(numberOfFeatures):
		distance[i]+=(newClientScaled[j] - kmObject.cluster_centers_[i,j])**2

print distance
print "The new client belongs to cluster number ", np.argmin(distance)










