import simplejson
import numpy
import scipy 
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from numpy import size
from numpy import transpose

with open('./newData/new-data/Customer_info.json', encoding = 'utf-8') as data_file:
	data = simplejson.load(data_file)

sampleNum = size(data["clients"])
infoMatrix = numpy.zeros((sampleNum,7), int)
validSampleNum = 0

for i in range(sampleNum):
	if "customerBaseExt" in data["clients"][i]:
		validSampleNum += 1
		birthDay = data["clients"][i]["customerBaseExt"]["birthday"]
		birthYear = int(birthDay[0:4])
		infoMatrix[i,0] = 2016 - birthYear

		gender = data["clients"][i]["customerBaseExt"]["gender"]
		if gender == "Male":
			infoMatrix[i,1] = 1
		else:
			infoMatrix[i,1] = 2

		if data["clients"][i]["customerMarriage"]["isMarried"] == "Married":
			infoMatrix[i,2] = 1
		else:
			infoMatrix[i,2] = 2

		educationString = data["clients"][i]["customerEducations"][0]["education"]
		if educationString == "Bachelor":
			infoMatrix[i,3] = 4
		elif educationString == "College":
			infoMatrix[i,3] = 3
		elif educationString == "Senior":
			infoMatrix[i,3] = 2
		else:
			infoMatrix[i,3] = 1

		totalMonthlyIncome = data["clients"][i]["customerAssetsIncome"]["monthlySalary"] + data["clients"][i]["customerAssetsIncome"]["monthlyIncomeOther"]
		if totalMonthlyIncome * 12 > data["clients"][i]["customerAssetsIncome"]["yearlyIncome"]:
			assetIncome = totalMonthlyIncome - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]
		else:
			assetIncome = data["clients"][i]["customerAssetsIncome"]["yearlyIncome"] / 12 - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]
		#assetIncome = data["clients"][i]["customerAssetsIncome"]["monthlySalary"] + data["clients"][i]["customerAssetsIncome"]["monthlyIncomeOther"] - data["clients"][i]["customerAssetsIncome"]["monthlyExpense"]
		#infoMatrix[i,4] = assetIncome

		infoMatrix[i,5] = data["clients"][i]["customerAssetsIncome"]["dependentNumber"]

	#print(data["clients"][i])

		if data["clients"][i]["customerAssetsVehicles"]:
			infoMatrix[i,6] = data["clients"][i]["customerAssetsVehicles"][0]["vehiclePurchasePrice"]

	#infoMatrix[i,1] =int(data["clients"][i]["customerBaseExt"]["householdRegisterAddrCity"])



validInfoMatrix = numpy.zeros((validSampleNum, 7), int)
j = 0
for i in range(sampleNum):
	if infoMatrix[i][0] != 0:
		validInfoMatrix[j] = infoMatrix[i]
		j += 1

#print(validInfoMatrix)
theta = numpy.ones(7, float)
numCluster = 4

def distortion(dataInput, thetaInput):
	optimized = dataInput * thetaInput
	return kmeans(optimized, numCluster, 10)[1]

def partialDeri(data, theta, i):
	theta_big = theta
	theta_big[i] = theta[i] + h
	dist_big = distortion(validInfoMatrix, theta_big)
	theta_small = theta
	theta_small[i] = theta[i] - h
	dist_small = distortion(validInfoMatrix, theta_small)
	return (dist_big - dist_small)/(2*h)

def update(data, theta):
	temp = numpy.zeros(size(theta))
	for i in range(7):
		temp[i] = theta[i] - alpha * partialDeri(validInfoMatrix, theta, i)
	for i in range(7):
		theta[i] = temp[i]

#print(validInfoMatrix)

iter = 1000
alpha = 0.0000005
h = 0.001
tempDist = 100
tempTheta = theta
gradIter = 20

for j in range(gradIter):
	theta = numpy.ones(7,float)
	for i in range(iter):
		update(validInfoMatrix, theta)
		print(distortion(validInfoMatrix, theta))
		if i >= 1 and distortion(validInfoMatrix, theta) < 10: break
	print(j)
	if distortion(validInfoMatrix, theta) < tempDist:
		tempDist = distortion(validInfoMatrix, theta)
		tempTheta = theta
		print(tempDist)

theta = tempTheta
print(theta)
centroids = kmeans(validInfoMatrix*theta, numCluster)[0]
print(centroids)

newClient = numpy.zeros(7, int)
'''newClient[0] = input("Age: ")
newClient[1] = input("Gender: ")
newClient[2] = input("Married: ")
newClient[3] = input("Education: ")
newClient[4] = input("Net income: ")
newClient[5] = input("Dependent: ")
newClient[6] = input("Vehicle: ")'''
newClient = newClient * theta

distance = numpy.zeros(numCluster, int)
for i in range(4):
	for j in range(numFeature):
		distance[i] += abs(newClient[j] - centroids[i][j])
	if i == 0:
		minDist = distance[i]
		index = i
	else:
		if distance[i] < minDist:
			minDist = distance[i]
			index = i

print(i)





