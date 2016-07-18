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
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder


data = []
numSample = 0
with open('../申请客户信息.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1
with open('../20160718.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1


infoMatrix = np.zeros((numSample, 0),)
label = np.zeros((numSample))
catagoricalColumns = []
numericalColumns = []
print ('now we have ', numSample,' samples')


for i in range(numSample):
	#0. Type applied
	validColumnsCount = 0
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][4] == '信优贷':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][4] == '信薪贷':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][4] == '信薪佳人贷':
		infoMatrix[i][validColumnsCount] = 2
	elif (data[i][4] == '薪期贷' or data[i][4]=='网薪期'):
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1
	
	#1. Duration applied
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][5] == '24':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][5] == '36':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][5] == '48':
		infoMatrix[i][validColumnsCount] = 2
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1
	

	#2.Amount applied
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][6])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1


	#3. Purpose of lending
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][7] == '消费':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][7] == '经营周转':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][7] == '其他':
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1
	
	#4.Maximun acceptable monthly payment
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][8])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1
	
	#5. Whether family members knew
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		catagoricalColumns.append(validColumnsCount)
	if data[i][9] == '是':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][9] == '否':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1

	#6. Type of residence
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
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
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1

	#7.Time lived in city
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][18])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1

	#8.Education background columns
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		numericalColumns.append(validColumnsCount)
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
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1

	#9Maritial status
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][21]=='已婚':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][21]=='未婚':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][21] == '离异':
		infoMatrix[i][validColumnsCount]=2
	else:
		infoMatrix[i][validColumnsCount]= np.nan
	validColumnsCount+=1

	#10.Gender 
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][22]=='男':
		infoMatrix[i][validColumnsCount]= 1
	elif data[i][22] == '女':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1

	#11Value of vehicle
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		numericalColumns.append(validColumnsCount)
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
		catagoricalColumns.append(validColumnsCount)
	if data[i][32]=='一般正式员工':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][32]=='中级管理人员':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][32]=='一般管理人员':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][32]=='派遣员工':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][32]=='高级管理人员':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][32] == '负责人':
		infoMatrix[i][validColumnsCount]=5
	else:
		infoMatrix[i][validColumnsCount]= np.nan
	validColumnsCount+=1


	#13	Type of working unit
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		catagoricalColumns.append(validColumnsCount)
	if data[i][33]=='机关事业单位':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][33]=='外资企业':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][33]=='私营企业':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][33]=='国有股份':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][33]=='合资企业':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][33]=='民营企业':
		infoMatrix[i][validColumnsCount]=5
	elif data[i][33] =='个体':
		infoMatrix[i][validColumnsCount]=6
	else:
		infoMatrix[i][validColumnsCount]= np.nan
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
		label[i] = 0
	elif data[i][39] == '薪期贷13':
		label[i] = 0
	elif data[i][39] == '薪期贷10':
		label[i] = 0
	elif data[i][39] == '薪期贷07':
		label[i] = 0
	else:
		label[i] = np.nan
	





#===Data imputation to be added below here===

#Impute catagorical data
imputerObjectFrequency = Imputer(missing_values='NaN', strategy='most_frequent',)
for i in catagoricalColumns:
	infoMatrix[:,i:i+1] = imputerObjectFrequency.fit_transform(infoMatrix[:,i:i+1])
label = imputerObjectFrequency.fit_transform(label.reshape(-1,1))




#Impute numerical data
imputerObjectMean = Imputer(missing_values='NaN', strategy='mean')
for i in numericalColumns:
	infoMatrix[:,i:i+1] = imputerObjectMean.fit_transform(infoMatrix[:,i:i+1])




#Perform one-hot encoding
encodingObject = OneHotEncoder(categorical_features = catagoricalColumns, sparse=False)
infoMatrix = encodingObject.fit_transform(infoMatrix)
#print (infoMatrix)




scaledMatrix = whiten(infoMatrix)




numTestCases = 900
#Count the number of 0123 in label
zeros = 0
ones = 0
twos = 0
threes = 0
for i in range(numTestCases,numSample):
	if (label[i] == 0):
		zeros +=1
	elif (label[i] == 1):
		ones+=1
	elif (label[i]==2):
		twos+=1
	else:
		threes+=1
print ('In the actual label, the number of 0 1 2 3 are ',zeros,'   ', ones, '  ', twos , '  ', threes, '  ', 'respectively')







#K-Nearest Neighbors classification
kneighborObject = KNeighborsClassifier(5)
kneighborObject.fit(scaledMatrix[0:numTestCases,:],np.ravel(label[0:numTestCases]))
print ('The training accuracy from KNN classification algorithm is: ', kneighborObject.score(scaledMatrix[numTestCases:numSample,:],np.ravel(label[numTestCases:numSample]))*100, '%')


#Support vector machine classification
svmClassWeight = {0:3,1:1,2:1.13,3:1}	#This suffices. Assigning sample weight has essentially the same effect on the result.
svmObject = svm.SVC(class_weight=svmClassWeight, probability = True)
svmObject.fit( scaledMatrix[0:numTestCases,:], np.ravel(label[0:numTestCases]))

correctPredictionsSvm = 0
zeros = 0
ones = 0
twos = 0
threes = 0
for i in range(numTestCases,numSample):
	if (svmObject.predict(scaledMatrix[i:i+1,:]) == 0):
		zeros +=1
	elif (svmObject.predict(scaledMatrix[i:i+1,:]) == 1):
		ones+=1
	elif (svmObject.predict(scaledMatrix[i:i+1,:])==2):
		twos+=1
	else:
		threes+=1
		print ('In this ', i,'th example, which was predicted to be 3, the label was actually', label[i])
	if (label[i]==0):
		print ('For',i,'th, the label is 0 but it\'s predicted to be ', svmObject.predict(scaledMatrix[i:i+1,:]) )
print ('The number of 0 1 2 3 are ',zeros,'   ', ones, '  ', twos , '  ', threes, '  ', 'respectively')
print ('The training accuracy from SVM learning algorithm is: ',svmObject.score( scaledMatrix[numTestCases:numSample,:], np.ravel(label[numTestCases:numSample])) *100, '%')




#Random forest classification
rfObject = RandomForestClassifier()
rfObject.fit(scaledMatrix[0:numTestCases,:],np.ravel(label[0:numTestCases]))
print ('The training accuracy from Random Forest algorithm is: ', 100*rfObject.score(scaledMatrix[numTestCases:numSample,:],np.ravel(label[numTestCases:numSample])), '%')




#Naive Bayes
nbObject = BernoulliNB()
nbObject.fit(scaledMatrix[0:numTestCases,:],np.ravel(label[0:numTestCases]))
print ('The training accuracy from Naive Bayes algorithm is: ', 100*nbObject.score(scaledMatrix[numTestCases:numSample,:],np.ravel(label[numTestCases:numSample])) , '%')




print ('==End of program==')

