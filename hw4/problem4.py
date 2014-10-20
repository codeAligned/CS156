import random
import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def dot(list1, list2):
	assert(len(list1) == len(list2))
	result = 0
	for i in range(len(list1)):
		result += (list1[i] * list2[i])

	return result


def generatePoint():
	x = random.uniform(-1,1)
	y = random.uniform(-1,1)
	return (x,y)

def sign(value):
	if (value >= 0.0):
		return 1.0
	else:
		return -1.0

def sine():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)

	# Calculate the best a
	a = ((x1 * y1) + (x2 * y2)) / ((x1*x1) + (x2 * x2))
	print a
	
	t = np.arange(-1, 1.5, 0.5)
	"""
	xvalues = [x1, x2]
	yvalues = [y1,y2]
	plt.plot(t, a * t, 'bs-', xvalues, yvalues, ':rs')
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""
	aMiss = (a - 1.426) * (a - 1.426) * x1 * x1
	return a, aMiss

aCounter = 0
aVar = 0 
sqTotal = 0
for i in range(100000):
	(a, aMiss) = sine()
	aCounter += a
	aVar += aMiss
	for i in range(1000):
		x = random.uniform(-1,1)
		sq = (a*x - 1.426 * x) * (a*x - 1.426 * x)
		sqTotal += sq
print "AVG A:", aCounter / 100000
print "Var: ", aVar/100000
print "SQtotal: ", sqTotal / 100000000

# now, generate 1000 random points to determine the average squared error
# between g(x) = 1.426x and f(x)

aError = 0
for i in range(100000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	g1 = 1.426 * x1   # g(x) estimate
	aError += ((g1 - y1) * (g1 - y1))

print "AVG BIAS: ", aError / 100000

