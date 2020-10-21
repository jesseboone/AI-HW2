from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import pickle

# split a string into tokens based on a given character
def parse (str1, char) :
    l = str1.split(char)
    return l

# building inputs x and y
# Reading them in from x.csv and y.csv
xs = []
ys = []
Z = []
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

f = open('z.csv', 'r')
for line in f.readlines():
	for num in parse(line, ','):
		Z.append(float(num))
f.close()

size = len(xs)
# normalizing the inputs to 0-1 scale
# could use a scikit learn normalize funtion instead but this works for now
for i in range(size):
	xs[i] = (xs[i]+1.)/4.0
	ys[i] = (ys[i]+3.)/4.0

# paired input array (xs and ys)
XY = []

# used to build out new pairs for single input array
for i in range(size):
	for k in range(size):
		XY = XY + [[xs[i], ys[k]]]

XY_train, XY_test, Z_train, Z_test = train_test_split(XY, Z, test_size=0.5, random_state=42)
Z_predicted = []

model = MLPRegressor(hidden_layer_sizes=(15,7), solver = 'adam', learning_rate='invscaling', random_state=42, max_iter=500, early_stopping=True)
model.fit(XY_train, Z_train)

filename = 'mlpModel.sav'
pickle.dump(model, open(filename, 'wb'))

for xy in XY_test:
	Z_predicted.append(model.predict([xy])[0])

print('r2_score on test data: ')
print(r2_score(Z_test, Z_predicted))

# f = open('output.txt', 'w')
# for i in range(size*size):
# 	s = str(model.predict([XY[i]]))[1:-1]
# 	f.write(s + "\n")
# f.close()

# result = []
# for xy in XY:
# 	result.append(model.predict([xy])[0])

# print('r2_score: ')
# print(r2_score(Z, result))
