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
			if (line[3] == '直销团队'):
				data.append(line)
				numSample += 1

infoMatrix = np.zeros((numSample, 0),)
label = np.zeros(numSample)
isApproved = np.zeros(numSample)
catagoricalColumns = []
numericalColumns = []
print ('now we have ', numSample,' samples')

