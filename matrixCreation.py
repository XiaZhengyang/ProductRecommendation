import simplejson
import numpy as np
import scipy 
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from numpy import array
from pprint import pprint
np.seterr(divide='ignore', invalid='ignore')

with open('../clientInformation.json') as data_file:
	data = simplejson.load(data_file)


infoMatrix = np.zeros((4,7),int)


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


print infoMatrix

whitened = whiten(infoMatrix)
for i in range (0,4):
	whitened[i,1] = 0;
	whitened[i,2] = 1;

print whitened

print kmeans(whitened,2)
