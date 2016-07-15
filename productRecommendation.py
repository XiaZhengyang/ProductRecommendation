import csv
import random
import numpy as np
import scipy 
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from numpy import size
import numpy.matlib
from sklearn.cluster import KMeans
np.set_printoptions(precision=1,threshold=1000000)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

with open('../申请客户信息.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	data = []
	numSample = 0
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1


infoMatrix = np.zeros((numSample, 0))
label = np.zeros((numSample),int)



for i in range(numSample):
	#0. Type applied
	validColumnsCount = 0
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][4] == '信优贷':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][4] == '信薪贷':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][4] == '信薪佳人贷':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][4] == '薪期贷':
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1
	
	#1. Duration applied
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][5] == '24':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][5] == '36':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][6] == '48':
		infoMatrix[i][validColumnsCount] = 2
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1
	

	#2.Amount applied
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][6])
	except:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1


	#3. Purpose of lending
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][7] == '消费':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][7] == '经营周转':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][7] == '其他':
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1
	
	#4.Maximun acceptable monthly payment
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][8])
	except:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1
	
	#5. Whether family members knew
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
	if data[i][9] == '是':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][9] == '否':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1

	#6. Type of residence
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][17] == '无按揭购房':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][17] == '商业按揭房':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][17] == '公积金按揭购房':
		infoMatrix[i][validColumnsCount] = 3
	elif data[i][17] == '自建房':
		infoMatrix[i][validColumnsCount] = 4
	elif data[i][17] == '单位住房':
		infoMatrix[i][validColumnsCount] = 5
	elif data[i][17] == '亲属住房':
		infoMatrix[i][validColumnsCount] = 6
	elif data[i][17] == '租用':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1

	#7.Time lived in city
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][18])
	except:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1

	#8.Education background columns
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
	if data[i][19]=='大学本科':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][19]=='高中及中专':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][19]=='大专':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][19]=='硕士':
		infoMatrix[i][validColumnsCount] = 3
	elif data[i][19] =='初中及以下':
		infoMatrix[i][validColumnsCount] = 4
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1

	#9Maritial status
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][21]=='已婚':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][21]=='未婚':
		infoMatrix[i][validColumnsCount]=1
	else:
		infoMatrix[i][validColumnsCount]=-1
	validColumnsCount+=1

	#10.Gender 
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
	if data[i][22]=='男':
		infoMatrix[i][validColumnsCount]= 1
	elif data[i][22] == '女':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = -1
	validColumnsCount+=1

	#11Value of vehicle
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
	if data[i][25]=='':
		infoMatrix[i][validColumnsCount]=0
	else:
		infoMatrix[i][validColumnsCount]=float(data[i][25])
	validColumnsCount+=1


	#12.Job and working company
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
	if data[i][31]=='一般正式员工':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][31]=='中级管理人员':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][31]=='一般管理人员':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][31]=='派遣员工':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][31]=='高级管理人员':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][31] == '负责人':
		infoMatrix[i][validColumnsCount]=5
	else:
		infoMatrix[i][validColumnsCount]=-1
	validColumnsCount+=1


	#13	Type of working unit
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
	if data[i][32]=='机关事业单位':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][32]=='外资企业':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][32]=='私营企业':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][32]=='国有股份':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][32]=='合资企业':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][32]=='民营企业':
		infoMatrix[i][validColumnsCount]=5
	elif data[i][32] =='个体':
		infoMatrix[i][validColumnsCount]=6
	else:
		infoMatrix[i][validColumnsCount]=-1
	validColumnsCount+=1

	if data[i][39] == '信优贷23':
		label[i] = 1
	elif data[i][39] == '信薪贷25':
		label[i] = 2
	elif data[i][39] == '信薪贷23':
		label[i] = 2
	elif data[i][39] == '信优贷19':
		label[i] = 1
	elif data[i][39] == '信薪佳人贷21':
		label[i] = 3
	elif data[i][39] == '信优贷17_A11':
		label[i] = 1
	elif data[i][39] == '信优贷21':
		label[i] = 1
	elif data[i][39] == '信薪贷27':
		label[i] = 2
	elif data[i][39] == '薪期贷17':
		label[i] = 4
	elif data[i][39] == '薪期贷13':
		label[i] = 4
	elif data[i][39] == '薪期贷10':
		label[i] = 4
	elif data[i][39] == '薪期贷07':
		label[i] = 4
	else:
		label[i]=5
	


#===Data imputation to be added below here===

print(infoMatrix)
print (label)
'''scaledMatrix = whiten(infoMatrix)



#K-Nearest Neighbors classification
kneighborObject = KNeighborsClassifier(5)
kneighborObject.fit(scaledMatrix[0:321,:],label[0:321])
correctPredictions = 0
for i in range(321,numSample):
	if  (kneighborObject.predict(scaledMatrix[i:i+1,:])==label[i]):
		correctPredictions+=1
print ('The training accuracy from KNN classification algorithm is: ', (correctPredictions/(numSample-321))*100, '%')


#Support vector machine classification
svmObject = svm.SVC()
svmObject.fit(scaledMatrix[0:321,:],label[0:321])
correctPredictionsSvm = 0
for i in range(321,numSample):
	if (svmObject.predict(scaledMatrix[i:i+1,:]) == label[i]):
		correctPredictionsSvm +=1
print ('The training accuracy from SVM classification algorithm is: ', 100*correctPredictionsSvm/(numSample-321), '%')



#Random forest classification
rfObject = RandomForestClassifier()
rfObject.fit(scaledMatrix[0:321,:],label[0:321])
correctPredictionsRf = 0
#print (rfObject.predict(infoMatrix[321:numSample,:]))
for i in range(321,numSample):
	if (rfObject.predict(scaledMatrix[i:i+1,:])==label[i]):
		correctPredictionsRf +=1
print ('The training accuracy from Random Forest algorithm is: ', 100*correctPredictionsRf/(numSample-321), '%')

#Naive Bayes
nbObject = BernoulliNB()
nbObject.fit(scaledMatrix[0:321,:],label[0:321])
correctPredictionsNb = 0
sameWithSvmCount = 0
for i in range(321,numSample):
	if (nbObject.predict(scaledMatrix[i:i+1,:]) == label[i]):
		correctPredictionsNb +=1
	if (nbObject.predict(scaledMatrix[i:i+1,:]) == svmObject.predict(scaledMatrix[i:i+1,:])):
		sameWithSvmCount +=1
print ('The training accuracy from Naive Bayes algorithm is: ', 100*correctPredictionsNb/(numSample-321), '%')
print ('Similarity between naive bayes and SVM is: ', 100*sameWithSvmCount/(numSample-321),'%')



print ('==End of program==')'''


'''
scaledMatrix = whiten(infoMatrix)
scalingCoefficient = np.zeros((numFeature,1),)
for i in range(numFeature):
	j =0 			#traverse all training examples until a non-zero number if found
	while (infoMatrix[j,i]==0):
		j=j+1
	scalingCoefficient[i] = scaledMatrix[j,i]/infoMatrix[j,i]	
#print (scalingCoefficient)
#print (scaledMatrix)


#Perform optimization
kmObject = KMeans(12,n_init=200)
kmObject.fit(scaledMatrix)
print (kmObject.labels_, kmObject.inertia_)'''












'''sampleNum = size(data["clients"])
infoMatrix = numpy.zeros((sampleNum,7), float)
validSampleNum = 0
numFeature = 7
>>>>>>> Stashed changes
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

'''
