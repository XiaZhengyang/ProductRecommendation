import csv
import numpy
import scipy
from numpy import size
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
numpy.set_printoptions(precision=1,threshold=1000000)

with open('客户数据3.csv', encoding='gbk') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	next(reader)
	data = []
	numSample = 0
	for line in reader:
		if not line in data:
			data.append(line)
			numSample += 1

numTrain = 1000
infoMatrix = numpy.ones((numSample, 0))
label = numpy.zeros(numSample)
catagoricalColumns = []
numericalColumns = []

for i in range(numSample):
	
	validColumnsCount = 0
	
	#0 Age
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	if data[i][2]=='':
		infoMatrix[i][validColumnsCount]=0
	else:
		infoMatrix[i][validColumnsCount]=116-int(data[i][2][-2:])
	validColumnsCount+=1

	#1. Type applied
	if i == 0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][5] == '信优贷':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][5] == '信薪贷':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][5] == '信薪佳人贷':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][5] == '薪期贷':
		infoMatrix[i][validColumnsCount] = 3
	elif data[i][5] == '网薪期':
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount += 1
	
	#2. Duration applied
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][6] == '24':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][6] == '36':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][6] == '48':
		infoMatrix[i][validColumnsCount] = 2
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1
	

	#3. Amount applied
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][7])
	except:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1


	#4. Purpose of lending
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][8] == '消费':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][8] == '经营周转':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][8] == '个人资金周转':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][8] == '其他':
		infoMatrix[i][validColumnsCount] = 3
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1
	
	#5. Maximun acceptable monthly payment
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][9])
	except:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1
	
	#6. Whether family members knew
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][10] == '是':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][10] == '否':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	#7. Type of residence
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][18] == '无按揭购房':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][18] == '商业按揭房':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][18] == '公积金按揭购房':
		infoMatrix[i][validColumnsCount] = 3
	elif data[i][18] == '自建房':
		infoMatrix[i][validColumnsCount] = 4
	elif data[i][18] == '单位住房':
		infoMatrix[i][validColumnsCount] = 5
	elif data[i][18] == '亲属住房':
		infoMatrix[i][validColumnsCount] = 6
	elif data[i][18] == '租用':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	#8. Time lived in city
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	try:
		infoMatrix[i][validColumnsCount] = float(data[i][19])
	except:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	#9. Education background columns
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][20]=='大学本科':
		infoMatrix[i][validColumnsCount] = 0
	elif data[i][20]=='高中及中专':
		infoMatrix[i][validColumnsCount] = 1
	elif data[i][20]=='大专':
		infoMatrix[i][validColumnsCount] = 2
	elif data[i][20]=='硕士':
		infoMatrix[i][validColumnsCount] = 3
	elif data[i][20] =='初中及以下':
		infoMatrix[i][validColumnsCount] = 4
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	#10. Maritial status
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][22]=='已婚':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][22]=='未婚':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][22] == '离异':
		infoMatrix[i][validColumnsCount]=2
	else:
		infoMatrix[i][validColumnsCount]= numpy.nan
	validColumnsCount+=1

	#11. Gender 
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][23]=='男':
		infoMatrix[i][validColumnsCount]= 1
	elif data[i][23] == '女':
		infoMatrix[i][validColumnsCount] = 0
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	#12. Value of vehicle
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		numericalColumns.append(validColumnsCount)
	if data[i][26]=='':
		infoMatrix[i][validColumnsCount]=0
	else:
		infoMatrix[i][validColumnsCount]=float(data[i][26])
	validColumnsCount+=1


	#13. Job and working company
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][33]=='一般正式员工':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][33]=='中级管理人员':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][33]=='一般管理人员':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][33]=='派遣员工':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][33]=='高级管理人员':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][33] == '负责人':
		infoMatrix[i][validColumnsCount]=5
	else:
		infoMatrix[i][validColumnsCount]= numpy.nan
	validColumnsCount+=1


	#14. Type of working unit
	if i==0:
		tempMatrix = numpy.zeros((numSample, numpy.size(infoMatrix,1)+1))
		tempMatrix[:,:-1] = infoMatrix
		infoMatrix.resize(numpy.size(tempMatrix,0),numpy.size(tempMatrix,1))
		infoMatrix = tempMatrix
		catagoricalColumns.append(validColumnsCount)
	if data[i][34]=='机关事业单位':
		infoMatrix[i][validColumnsCount]=0
	elif data[i][34]=='外资企业':
		infoMatrix[i][validColumnsCount]=1
	elif data[i][34]=='私营企业':
		infoMatrix[i][validColumnsCount]=2
	elif data[i][34]=='国有股份':
		infoMatrix[i][validColumnsCount]=3
	elif data[i][34]=='合资企业':
		infoMatrix[i][validColumnsCount]=4
	elif data[i][34]=='民营企业':
		infoMatrix[i][validColumnsCount]=5
	elif data[i][34] =='个体':
		infoMatrix[i][validColumnsCount]=6
	else:
		infoMatrix[i][validColumnsCount] = numpy.nan
	validColumnsCount+=1

	
	#Final type
	if data[i][40] == '信优贷23':
		label[i] = 1
	elif data[i][40] == '信薪贷25':
		label[i] = 2
	elif data[i][40] == '信薪贷23':
		label[i] = 2
	elif data[i][40] == '信优贷19':
		label[i] = 1
	elif data[i][40] == '信薪佳人贷21':
		label[i] = 3
	elif data[i][40] == '信优贷17_A11':
		label[i] = 1
	elif data[i][40] == '信优贷21':
		label[i] = 1
	elif data[i][40] == '信薪贷27':
		label[i] = 2
	elif data[i][40] == '薪期贷17':
		label[i] = 0
	elif data[i][40] == '薪期贷13':
		label[i] = 0
	elif data[i][40] == '薪期贷10':
		label[i] = 0
	elif data[i][40] == '薪期贷07':
		label[i] = 0
	else:
		label[i] = 4
	


#===Data imputation to be added below here===

#Impute catagorical data
imputerObjectFrequency = Imputer(missing_values='NaN', strategy='most_frequent',)
for i in catagoricalColumns:
	infoMatrix[:,i:i+1] = imputerObjectFrequency.fit_transform(infoMatrix[:,i:i+1])

#Impute numerical data
imputerObjectMean = Imputer(missing_values='NaN', strategy='mean')
for i in numericalColumns:
	infoMatrix[:,i:i+1] = imputerObjectMean.fit_transform(infoMatrix[:,i:i+1])


#Perform one-hot encoding
encodingObject = OneHotEncoder(categorical_features = catagoricalColumns, sparse=False)
infoMatrix = encodingObject.fit_transform(infoMatrix)
stdInfoMatrix = scale(infoMatrix)

dict = {0:3, 1:1, 2:1.3, 3:1}
svmObject = svm.SVC(class_weight=dict, probability=True)
svmObject.fit(stdInfoMatrix[:numTrain],label[:numTrain])
correctPredictionCount = 0
proba = svmObject.predict_proba(stdInfoMatrix[numTrain:])
for i in range(numSample-numTrain):
	tempLabel = 0
	for j in range(4):
		if proba[i][j] > proba[i][tempLabel]:
			tempLabel = j
	if tempLabel == label[i+numTrain]:
		correctPredictionCount += 1

print(correctPredictionCount/(numSample-numTrain)*100, '%', sep = '')