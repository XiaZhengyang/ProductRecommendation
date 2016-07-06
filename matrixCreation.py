<<<<<<< HEAD
import simplejson
import numpy as np
from pprint import pprint

with open('./clientInformation.json') as data_file:
	data = simplejson.load(data_file)


infoMatrix = np.zeros((4,7),int)

infoMatrix[3,6]=5

for i in range (0,4):
	infoMatrix[i,0]= int(data["clients"][i]["customerBaseExt"]["birthday"])
	infoMatrix[i,1] =int(data["clients"][i]["customerBaseExt"]["householdRegisterAddrCity"])
	if data["clients"][i]["customerMarriage"]["isMarried"] == "Married":
		infoMatrix[i,2] ="1"
	else:
		infoMatrix[i,3] = "0"

print infoMatrix
=======
print "test"
>>>>>>> origin/master
