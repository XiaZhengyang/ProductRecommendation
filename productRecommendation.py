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
numFeature = 45
infoMatrix = numpy.zeros((numSample, numFeature))
label = numpy.zeros((numSample),int)


for i in range(numSample):
	#0-3. Type applied
	if data[i][4] == '信优贷':
		infoMatrix[i][0] = 1
	elif data[i][4] == '信薪贷':
		infoMatrix[i][1] = 1
	elif data[i][4] == '信薪佳人贷':
		infoMatrix[i][2] = 1
	else:#Exceptions: Xinqidai & one instance of Luxinyou
		infoMatrix[i][3] = 1
	
	#4-6. Duration applied
	if data[i][5] == '24':
		infoMatrix[i][4] = 1
	elif data[i][5] == '48':
		infoMatrix[i][5] = 1
	else:
		infoMatrix[i][6] = 1
	
	#7.Amount applied
	infoMatrix[i][7] = float(data[i][6])


	#8-11. Purpose of lending
	if data[i][7] == '消费':
		infoMatrix[i][8] = 1
	elif data[i][7] == '经营周转':
		infoMatrix[i][9] = 1
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][10] = 1
	else:#Exceptions: Else
		infoMatrix[i][11] = 1
	
	#12.Maximun acceptable monthly payment
	try:
		infoMatrix[i][12] = float(data[i][8])
	except:
		infoMatrix[i][12] = 0
	
	#13. Whether family members knew
	if data[i][9] == '是':
		infoMatrix[i][13] = 1

	#14-21. Type of residence
	if data[i][17] == '无按揭购房':
		infoMatrix[i][14] = 1
	elif data[i][17] == '商业按揭房':
		infoMatrix[i][15] = 1
	elif data[i][17] == '公积金按揭购房':
		infoMatrix[i][16] = 1
	elif data[i][17] == '自建房':
		infoMatrix[i][17] = 1
	elif data[i][17] == '单位住房':
		infoMatrix[i][18] = 1
	elif data[i][17] == '亲属住房':
		infoMatrix[i][19] = 1
	else:
		infoMatrix[i][20] = 1

	#22.Time lived in city
	try:
		infoMatrix[i][21] = float(data[i][18])
	except:
		infoMatrix[i][21] = 0

	#22-26.Education background columns
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

	#27-29Maritial status
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
		infoMatrix[i][31]=float(data[i][25])


	#32-37Job and working company
	if data[i][32]=='一般正式员工':
		infoMatrix[i][32]=1
	elif data[i][32]=='中级管理人员':
		infoMatrix[i][33]=1
	elif data[i][32]=='一般管理人员':
		infoMatrix[i][34]=1
	elif data[i][32]=='派遣员工':
		infoMatrix[i][35]=1
	elif data[i][32]=='高级管理人员':
		infoMatrix[i][36]=1
	else:
		infoMatrix[i][37]=1


	#39-44	Type of working unit
	if data[i][33]=='机关事业单位':
		infoMatrix[i][39]=1
	elif data[i][33]=='外资企业':
		infoMatrix[i][39]=1
	elif data[i][33]=='私营企业':
		infoMatrix[i][40]=1
	elif data[i][33]=='国有股份':
		infoMatrix[i][41]=1
	elif data[i][33]=='合资企业':
		infoMatrix[i][42]=1
	elif data[i][33]=='民营企业':
		infoMatrix[i][43]=1
	else:
		infoMatrix[i][44]=1

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
	

#print(infoMatrix)
#print (label)
scaledMatrix = whiten(infoMatrix)



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



print ('==End of program==')
