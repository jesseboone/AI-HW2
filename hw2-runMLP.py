from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import pickle
import numpy as np

# split a string into tokens based on a given character
def parse (str1, char) :
    l = str1.split(char)
    return l

# building inputs x and y
# Reading them in from x.csv and y.csv
xs = []
ys = []
size = 0

f = open('x.csv', 'r')
for line in f.readlines():
	for num in parse(line, ','):
		xs.append(float(num))
f.close()

f = open('y.csv', 'r')
for line in f.readlines():
	for num in parse(line, ','):
		ys.append(float(num))
f.close()

size = len(xs)
# normalizing the inputs to 0-1 scale
# could/should use a scikit learn normalize funtion instead but this works for now
for i in range(size):
	xs[i] = (xs[i]+1.)/4.0
	ys[i] = (ys[i]+3.)/4.0

# paired input array (xs and ys)
XY = []

# used to build out new pairs for single input array
for i in range(size):
	for k in range(size):
		XY = XY + [[xs[i], ys[k]]]

# load model from buildMLP
filename = 'mlpModel.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# use model to predict output Z
result = []
for xy in XY:
	result.append(loaded_model.predict([xy])[0])

# turn into a numpy array to reshape
# reshape into correct size (30x30 on example)
result = np.array(result)
result = result.reshape(size, size)

# super handy dandy little function tht automatically formats a numpy array into a csv...wish I had found this 2 hours ago - 11:44, 10/20/2020
np.savetxt('z-predicted.csv', result, delimiter=',', fmt='%f')
