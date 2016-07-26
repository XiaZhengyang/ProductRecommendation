import csv
import random
import numpy as np
import scipy 
from numpy import size
import numpy.matlib
np.set_printoptions(precision=1,threshold=1000000)
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


data = []
numSample = 0

with open('../20160722.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	for line in reader:
		if not line in data:
			if (line[3] == '直销团队' and line[42]!='' and line[43]!='' and line[50]!=''):
				data.append(line)
				numSample += 1

infoMatrix = np.zeros((numSample, 0),)
label = np.zeros(numSample)
isApproved = np.zeros(numSample)
catagoricalColumns = []
numericalColumns = []
print ('now we have ', numSample,' samples')


possibleLabels = []
for i in range(numSample):
	validColumnsCount=0

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


	#Payment of accu fund
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][42])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1


	#Base of accu fund
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][43])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1


	#Averave monthly salary
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix	
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][50])
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1



	#Labels
	if (data[i][39][:3] not in possibleLabels):
		possibleLabels.append(data[i][39][:3])
	label[i] = possibleLabels.index(data[i][39][:3])



#===Data preprocessing below===

#Impute catagorical data
imputerObjectFrequency = Imputer(missing_values='NaN', strategy='most_frequent')
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

#Scaling
scalingObject = StandardScaler()
scaledMatrix = scalingObject.fit_transform(infoMatrix)

print (label)

#Support vector machine classification
classWeight = {2:1.5,4:0.5,6:2,7:2.5,8:1.4,9:1.6}
svmObject_product = svm.SVC(C = 1.2 ,probability = True)
svmScores_product = cross_validation.cross_val_score(svmObject_product,scaledMatrix,np.ravel(label),cv=4)
print("Accuracy(SVM for product): %0.4f (+/- %0.3f)" % (svmScores_product.mean(), svmScores_product.std() * 2))



