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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


data = []
numSample = 0

with open('../20160719.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	for line in reader:
		if not line in data:
			if (line[39][:3] == '信优贷' or line[39][:3] =='信薪贷'):
				data.append(line)
				numSample += 1

infoMatrix = np.zeros((numSample, 0),)
label = np.zeros(numSample)
isApproved = np.zeros(numSample)
catagoricalColumns = []
numericalColumns = []
print ('now we have ', numSample,' samples')


for i in range(numSample):

	validColumnsCount = 0
	#Age
	if i==0:
		tempMatrix = np.zeros(( np.size(infoMatrix,0),np.size(infoMatrix,1)+1 ))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(np.size(tempMatrix,0),np.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = (2016-float(data[i][42][-4:]))
	except:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1



	#0. Type applied
	'''if i==0:
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
		numericalColumns.append(validColumnsCount)
	if data[i][7] == '消费':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][7] == '经营周转':
		infoMatrix[i][validColumnsCount] = 10000
	elif data[i][7] == '个人资金周转':
		infoMatrix[i][validColumnsCount] = 100
	elif data[i][7] == '其他':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = np.nan
	validColumnsCount+=1'''
	
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
	'''if i==0:
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
	validColumnsCount+=1'''

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
		catagoricalColumns.append(validColumnsCount)
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

	#9Marital status
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



	#Suitable product
	if data[i][39] == '信优贷23':
		label[i] = 1
	elif data[i][39] == '信薪贷25':
		label[i] = 2
	elif data[i][39] == '信薪贷23':
		label[i] = 3
	elif data[i][39] == '信优贷19':
		label[i] = 4
	elif data[i][39] == '信薪佳人贷21':
		label[i] = 5
	elif data[i][39] == '信优贷17_A11':
		label[i] = 6
	elif data[i][39] == '信优贷21':
		label[i] = 7
	elif data[i][39] == '信薪贷27':
		label[i] = 8
	elif data[i][39] == '薪期贷17':
		label[i] = 9
	elif data[i][39] == '薪期贷13':
		label[i] = 10
	elif data[i][39] == '薪期贷10':
		label[i] = 11
	elif data[i][39] == '薪期贷07':
		label[i] = 0
	else:
		label[i] = np.nan
	

	#Approval status
	if data[i][38] == '通过':
		isApproved[i] = 1
	else:
		isApproved[i] = 0



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




#====Actual Learning Processes====

#K-Nearest Neighbors classification
kneighborObject = KNeighborsClassifier(5)
kneighborScores = cross_validation.cross_val_score(kneighborObject,scaledMatrix,isApproved,cv=7)
print("Accuracy(Knn for approval): %0.4f (+/- %0.3f)" % (kneighborScores.mean(), kneighborScores.std() * 2))



#Support vector machine classification
classWeight = {2:1.5,4:0.5,6:2,7:2.5,8:1.4,9:1.6}
svmObject_product = svm.SVC(C = 1.2 ,probability = True)
svmScores_product = cross_validation.cross_val_score(svmObject_product,scaledMatrix[0:1600,:],np.ravel(label[0:1600]),cv=4)
print("Accuracy(SVM for product, first 1200 samples): %0.4f (+/- %0.3f)" % (svmScores_product.mean(), svmScores_product.std() * 2))

svmObject_product.fit(scaledMatrix,np.ravel(label))
labelsCount = np.zeros(12)
predictionsCount = np.zeros(12)
for i in range(numSample):
	labelsCount[int(label[i])]+=1
	predictionsCount[int( svmObject_product.predict(scaledMatrix[i:i+1,:]) )]+=1
print (labelsCount)
print (predictionsCount)






'''for i in range(numSample):
	print (svmObject_product.predict(scaledMatrix[i:i+1,:]))
encodeIndices = encodingObject.feature_indices_'''



'''input1 = int(input('Please input your age>>'))
input2 = int(input('Please input your maximum acceptable monthly payment>>'))
input3 = int(input('Please input your type of residence0-6>>'))
input4 = int(input('please input the number of years you have lived in this city>>'))
input5 = int(input('please input your education background0-4>>'))
input6 = int(input('please input your marital status0-2>>'))
input7 = int(input('please input your gender 1 for male>>'))
input8 = int(input('please input the value of your vehicle (if none input 0)>>'))
input9 = int(input('please input the job within your company0-5>>'))
input10 = int(input('please input the type of your company 0-6>>'))
encodedArray = np.zeros(np.size(scaledMatrix,1))
encodedArray[encodeIndices[0]+input3] = 1
encodedArray[encodeIndices[1]+input5] = 1
encodedArray[encodeIndices[2]+input6] = 1
encodedArray[encodeIndices[3]+input7] = 1
encodedArray[encodeIndices[4]+input9] = 1
encodedArray[encodeIndices[5]+input10] = 1
encodedArray[encodeIndices[6]] = input1
encodedArray[encodeIndices[6]+1] = input2
encodedArray[encodeIndices[6]+2] = input4
encodedArray[encodeIndices[6]+3] = input8
encodedArray = scalingObject.transform(encodedArray.reshape(1,-1))'''




#SVM classification
svmObject_approval = svm.SVC()
svmScores_approval = cross_validation.cross_val_score(svmObject_approval ,scaledMatrix,isApproved,cv=7)
print("Accuracy(SVM for approval): %0.4f (+/- %0.3f)" % (svmScores_approval.mean(), svmScores_approval.std() * 2))




#Random forest classification
rfObject_Approval = RandomForestClassifier()
rfScores = cross_validation.cross_val_score(rfObject_Approval ,scaledMatrix,isApproved,cv=7)
print("Accuracy(RF for approval): %0.4f (+/- %0.3f)" % (rfScores.mean(), rfScores.std() * 2))



#Naive Bayes
nbObject_Approval = BernoulliNB()
nbScores = cross_validation.cross_val_score(nbObject_Approval,scaledMatrix,isApproved,cv=7)
print("Accuracy(NB for approval): %0.4f (+/- %0.3f)" % (nbScores.mean(), nbScores.std() * 2))






print ('==End of program==')

