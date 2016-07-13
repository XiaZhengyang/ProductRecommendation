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
numpy.set_printoptions(precision=1,threshold=1000000)

with open('申请客户信息.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	data = []
	numSample = 0
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1
numFeature = 100
infoMatrix = numpy.zeros((numSample, numFeature))
label = numpy.zeros(13)
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
	elif data[i][5] == 48:
		infoMatrix[i][7] = 1
	else:
		infoMatrix[i][6] = 1
	infoMatrix[i][8] = data[i][6]
	if data[i][7] == '消费':
		infoMatrix[i][8] = 1
	elif data[i][7] == '经营周转':
		infoMatrix[i][9] = 1
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][10] = 1
	else:
		infoMatrix[i][11] = 1
	try:
		infoMatrix[i][12] = float(data[i][8])
	except:
		infoMatrix[i][12] = 0
	if data[i][9] == '是':
		infoMatrix[i][13] = 1
	if data[i][17] == '无按揭购房':
		infoMatrix[i][14] = 1
	elif data[i][17] == '商业按揭房':
		infoMatrix[i][15] = 1
	elif data[i][17] == '公积金按揭房':
		infoMatrix[i][16] = 1
	elif data[i][17] == '自建房':
		infoMatrix[i][17] = 1
	elif data[i][17] == '单位住房':
		infoMatrix[i][18] = 1
	elif data[i][17] == '亲属住房':
		infoMatrix[i][19] = 1
	else:
		infoMatrix[i][20] = 1
	try:
		infoMatrix[i][21] = float(data[i][18])
	except:
		infoMatrix[i][21] = 0

	#Education background columns
	if data[i][19]=='大学本科':
		infoMatrix[i][22] = 1
	elif data[i][19]=='高中及中专':
		infoMatrix[i][23] = 1
	elif data[i][19]=='大专':
		infoMatrix[i][24] = 1
	elif data[i][19]=='硕士':
		infoMatrix[i][25]=1
	else:
		infoMatrix[26] = 1

	#Maritial status
	if data[i][21]=='已婚':
		infoMatrix[i][27]=1
	elif data[i][21]=='未婚':
		infoMatrix[i][28]=1
	else:
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
	elif data[i][31]=='中级管理人员':
		infoMatrix[i][33]=1
	elif data[i][31]=='一般管理人员':
		infoMatrix[i][34]=1
	elif data[i][31]=='派遣员工':
		infoMatrix[i][35]=1
	elif data[i][31]=='高级管理人员':
		infoMatrix[i][36]=1
	else:
		infoMatrix[i][37]=1


	#38-44	Type of working unit
	if data[i][32]=='机关事业单位':
		infoMatrix[i][38]=1
	elif data[i][32]=='外资企业':
		infoMatrix[i][39]=1
	elif data[i][32]=='私营企业':
		infoMatrix[i][40]=1
	elif data[i][32]=='国有股份':
		infoMatrix[i][41]=1
	elif data[i][32]=='合资企业':
		infoMatrix[i][42]=1
	elif data[i][32]=='民营企业':
		infoMatrix[i][43]=1
	else:
		infoMatrix[i][44]=1

	if data[i][38] == '信优贷23':
		label[i] = 1
	elif data[i][38] == '信薪贷25':
		label[i] = 2
	elif data[i][38] == '信薪贷23':
		label[i] = 3
	elif data[i][38] == '信优贷19':
		label[i] = 4
	elif data[i][38] == '信薪佳人贷21':
		label[i] = 5
	elif data[i][38] == '信优贷17_A':
		label[i] = 6
	elif data[i][38] == '信优贷21':
		label[i] = 7
	elif data[i][38] == '信薪贷27':
		label[i] = 8
	elif data[i][38] == '薪期贷17':
		label[i] = 9
	elif data[i][38] == '信薪贷25':
		label[i] = 10
	elif data[i][38] == '信薪贷25':
		label[i] = 2
	elif data[i][38] == '信薪贷25':
		label[i] = 2
	

print(infoMatrix)



numCluster = 6
iter = 1000
alpha = 0.1


stdInfoMatrix = whiten(infoMatrix)
random.seed(version=2)
prototype = random.sample(range(sampleNum), numCluster)
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

	print(index)

