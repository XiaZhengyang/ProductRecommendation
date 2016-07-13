import csv
import scipy
import numpy as np
lineList = []

#Initial CSV File Reading
with open('../customerInfo-Chinese.csv',encoding='gbk') as f:
	reader = csv.reader(f,delimiter=',', quotechar=' ')
	next(reader)
	for row in reader:
		if row in lineList: 
			continue
		else:
			lineList.append(row)



#Creation of information matrix
numberOfFeatures = 7
validSampleCount = len(lineList)
totalNumberOfAttributes = len(lineList[0])
infoMatrix = np.empty((validSampleCount,totalNumberOfAttributes),str)
for i in range(validSampleCount):
	for j in range(totalNumberOfAttributes):
		infoMatrix[i,j] = lineList[i][j]
print (infoMatrix)

'''
#Scale the features to create a scaled matrix
scaledMatrix = whiten(infoMatrix)
scalingCoefficient = np.zeros((numberOfFeatures,1),)
for i in range(numberOfFeatures):
	j =0 			#traverse all training examples until a non-zero number is found
	while (infoMatrix[j,i]==0):
		j=j+1
	scalingCoefficient[i] = scaledMatrix[j,i]/infoMatrix[j,i]	
print scalingCoefficient
print scaledMatrix

'''

