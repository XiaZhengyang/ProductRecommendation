import csv
import random
import numpy
import scipy 
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from numpy import size
import numpy.matlib
numpy.set_printoptions(threshold='nan')

with open('../申请客户信息.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	data = []
	numSample = 0
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1
numFeature = 45
infoMatrix = numpy.zeros((numSample, numFeature))
for i in range(numSample):
	if data[i][4] == '信优贷':
		infoMatrix[i][0] = 1
	elif data[i][4] == '信薪贷':
		infoMatrix[i][1] = 1
	elif data[i][4] == '信薪佳人贷':
		infoMatrix[i][2] = 1
	else:
		infoMatrix[i][3] = 1
	if data[i][5] == 24:
		infoMatrix[i][4] = 1
	elif data[i][5] == 36:
		infoMatrix[i][5] = 1
	elif data[i][5] == 48:
		infoMatrix[i][6] = 1
	else:
		infoMatrix[i][7] = 1
	infoMatrix[i][8] = data[i][6]
	if data[i][7] == '消费':
		infoMatrix[i][8] = 1
	elif data[i][7] == '经营周转':
		infoMatrix[i][9] = 1
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][10] = 1
	else:
		infoMatrix[i][11] = 1
	#infoMatrix[i][12] = float(data[i][8])
	if data[i][9] == '是':
		infoMatrix[i][13] = 1


	#Education background columns
	if data[i][19]=='大学本科':
		infoMatrix[i][22] = 1
	if data[i][19]=='高中及中专':
		infoMatrix[i][23] = 1
	if data[i][19]=='大专':
		infoMatrix[i][24] = 1
	if data[i][19]=='硕士':
		infoMatrix[i][25]=1
	if data[i][19]=='初中及以下':
		infoMatrix[26] = 1

	#Maritial status
	if data[i][21]=='已婚':
		infoMatrix[i][27]=1
	if data[i][21]=='未婚':
		infoMatrix[i][28]=1
	if data[i][21]=='离异':
		infoMatrix[i][29]=1

	#30.Gender 
	if data[i][22]=='男':
		infoMatrix[i][30]=1

	#31Value of vehicle
	if data[i][25]=='':
		infoMatrix[i][31]=0
	else:
		infoMatrix[i][31]=float(data[2][25])


	#32-37Job and working company
	if data[i][31]=='一般正式员工':
		infoMatrix[i][32]=1
	if data[i][31]=='中级管理人员':
		infoMatrix[i][33]=1
	if data[i][31]=='一般管理人员':
		infoMatrix[i][34]=1
	if data[i][31]=='派遣员工':
		infoMatrix[i][35]=1
	if data[i][31]=='高级管理人员':
		infoMatrix[i][36]=1
	if data[i][31]=='负责人':
		infoMatrix[i][37]=1


	#38-44	Type of working unit
	if data[i][32]=='机关事业单位':
		infoMatrix[i][38]=1
	if data[i][32]=='外资企业':
		infoMatrix[i][39]=1
	if data[i][32]=='私营企业':
		infoMatrix[i][40]=1
	if data[i][32]=='国有股份':
		infoMatrix[i][41]=1
	if data[i][32]=='合资企业':
		infoMatrix[i][42]=1
	if data[i][32]=='民营企业':
		infoMatrix[i][43]=1
	if data[i][32]=='个体':
		infoMatrix[i][44]=1



print (infoMatrix)
	


'''sampleNum = size(data["clients"])
infoMatrix = numpy.zeros((sampleNum,7), float)
validSampleNum = 0
numFeature = 7
numCluster = 6
iter = 1000
alpha = 0.1

for i in range(sampleNum):
	if "customerBaseExt" in data["clients"][i]:
		validSampleNum += 1
		birthDay = data["clients"][i]["customerBaseExt"]["birthday"]
		birthYear = int(birthDay[0:4])
		infoMatrix[i,0] = 1916 - birthYear

		gender = data["clients"][i]["customerBaseExt"]["gender"]
		if gender == "Male":
			infoMatrix[i,1] = 0
		else:
			infoMatrix[i,1] = 1

		if data["clients"][i]["customerMarriage"]["isMarried"] == "Married":
			infoMatrix[i,2] = 0
		else:
			infoMatrix[i,2] = 1

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
		infoMatrix[i,4] = assetIncome

		infoMatrix[i,5] = data["clients"][i]["customerAssetsIncome"]["dependentNumber"]

	#print(data["clients"][i])

		if data["clients"][i]["customerAssetsVehicles"]:
			infoMatrix[i,6] = data["clients"][i]["customerAssetsVehicles"][0]["vehiclePurchasePrice"]

	#infoMatrix[i,1] =int(data["clients"][i]["customerBaseExt"]["householdRegisterAddrCity"])



validInfoMatrix = numpy.zeros((validSampleNum, 7), float)
j = 0
for i in range(sampleNum):
	if infoMatrix[i][0] != 0:
		validInfoMatrix[j] = infoMatrix[i]
		j += 1

stdInfoMatrix = whiten(validInfoMatrix)
label = numpy.zeros(validSampleNum, int)
random.seed(version=2)
prototype = random.sample(range(validSampleNum), numCluster)
centroids = stdInfoMatrix[prototype]
ptLabel = label[prototype]

def computeDistance(x,y):
	dist = 0
	for i in range(size(x)):
		dist += (x[i]-y[i])*(x[i]-y[i])
	return dist


def findNearestCentroid(index):
	minDist = computeDistance(stdInfoMatrix[index], centroids[0])
	minIndex = 0
	for i in range(numCluster):
		tempDist = computeDistance(stdInfoMatrix[index], centroids[i])
		if tempDist < minDist:
			minDist = tempDist
			minIndex = i
	return minIndex

def close(cIndex, sIndex):
	for i in range(numFeature):
		centroids[cIndex][i] += alpha * (stdInfoMatrix[sIndex][i] - centroids[cIndex][i])

def far(cIndex, sIndex):
	for i in range(numFeature):
		centroids[cIndex][i] -= alpha * (stdInfoMatrix[sIndex][i] - centroids[cIndex][i])

for i in range(iter):
	sampleIndex = random.choice(range(validSampleNum))
	centroidIndex = findNearestCentroid(sampleIndex)
	if label[sampleIndex] == label[centroidIndex]:
		close(centroidIndex, sampleIndex)
	else:
		far(centroidIndex, sampleIndex)

print(centroids)


centroids = kmeans(stdInfoMatrix, numCluster, 500)[0]
print(centroids)


while input('end? ') != 'yes':
	newClient = numpy.zeros(numFeature, int)
	newClient[0] = input("Age: ")
	newClient[1] = input("Gender: ")
	newClient[2] = input("Married: ")
	newClient[3] = input("Education: ")
	newClient[4] = input("Net Income: ")
	newClient[5] = input("Dependent: ")
	newClient[6] = input("Vehicle: ")

	distance = numpy.zeros(numCluster, float)
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

	print(index)'''


