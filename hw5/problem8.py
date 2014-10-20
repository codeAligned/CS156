import random
import math
from numpy import *
from numpy.linalg import *
from numpy.matlib import *
import matplotlib.pyplot as plt


def dotProduct(list1, list2):
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

# returns False if we have a missclassifed points
def checkMisclassified(points, weights, results):
	for i in range(len(points)):
		result = sign(dotProduct(weights, list(points[i])))
		if (result != results[i]):
			return False
	return True

def calcEout(weights, points, results):

	hypSlope = -1 * weights[1] / weights[2]
	hypIntercept = -1 * weights[0] / weights[2]

	#t = arange(-1, 1.5, 0.5)
	#plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'rs-', xA, yA, ':rs', xB, yB, ':gs')
	#plt.axis( [-1, 1, -1, 1])
	#plt.show()

	sumTotal = 0.0
	for i in range(100):
		(x1, y1) = generatePoint()
		# y = mx + b
		x = [1, x1, y1]
		#yval = hypSlope * x1 + hypIntercept
		yval = x[0] * weights[0] + x[1] * weights[1] + x[2] * weights[2]
		if yval > 0:
			result = 1
		else:
			result = -1

		cross = math.log(1 + math.exp(-result * dotProduct(weights, x)))
		"""
		if cross > 2:
			print cross
			print 'R: ', result
			print weights
			print x
		"""
		sumTotal += cross
	print "EOUT:  ", sumTotal / 100
	return sumTotal / 100



def regression(N, n):
	(x1,y1) = generatePoint()
	(x2,y2) = generatePoint()

	rise = y2 - y1
	run = x2 - x1

	slope = rise / run

	# y = mx + b
	#b = y - mx
	b = y1 - slope * x1

	# Now, we have y = (slope) * x + b
	#print "y = ", slope, "x + ", b
	xvalues = range(N)
	yvalues = range(N)

	points = range(N)
	for i in range(N):
		(x,y) = generatePoint()
		points[i] = (1, x,y)
		xvalues[i] = x
		yvalues[i] = y

	xA = []
	yA = []
	xB = []
	yB = []
	results = range(N)
	for j in range(N):
		point = points[j]
		x = point[1]
		y = point[2]

		yVal = slope * x + b
		if (y > yVal):
			results[j] = 1
			xA.append(x)
			yA.append(y)
		else:
			results[j] = -1
			xB.append(x)
			yB.append(y)

	

	counter = 0
	weights = [0,0,0]
	while True:
		orig = weights[:]
		#orig0 = weights[0]
		#orig1 = weights[1]
		#orig2 = weights[2]
		permutation = range(100)
		random.shuffle(permutation)
		#print permutation
		counter += 1

		for index in permutation:
			#print "I: " , index
			x = points[index]
			y = results[index]
			#print "I: ", index, "  weights! ", weights
			error1 = -(y * x[0]) / (1 + math.exp(y * dotProduct(weights, x)))
			error2 = -(y * x[1]) / (1 + math.exp(y * dotProduct(weights, x)))
			error3 = -(y * x[2]) / (1 + math.exp(y * dotProduct(weights, x)))
			error = [0.01 * error1, 0.01 * error2, 0.01 * error3]
			#weights = weights - error
			weights[0] = weights[0] - error[0]
			weights[1] = weights[1] - error[1]
			weights[2] = weights[2] - error[2]
		# now, compare original weights. did we break out?
		#print  "ORIG: " , originalWeights
		#print "ORIG: ", orig0, orig1, orig2
		#print "WEIGHTS: ", weights
		length = (orig[0] - weights[0]) * (orig[0] - weights[0])
		length += (orig[1] - weights[1]) * (orig[1] - weights[1])
		length += (orig[2] - weights[2]) * (orig[2] - weights[2])
		sq = math.sqrt(length)

		if sq < 0.01:
			#print "COUNT: ", counter
			#print "LENGTH: ", sq
			#print weights

			"""
			hypSlope = -1 * weights[1] / weights[2]
			hypIntercept = -1 * weights[0] / weights[2]
			s = arange(-1, 1.5, 0.5)

			t = arange(-1, 1.5, 0.5)
			plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'rs-', xA, yA, ':rs', xB, yB, ':gs')
			plt.axis( [-1, 1, -1, 1])
			plt.show()
			"""
			
			eOut = calcEout(weights, points, results)


			return (weights, eOut, counter)
		else:
			a = 1
			#print sq


N = 100
n = 0.01


eOutCounter = 0.0
iterationsCounter = 0.0
for i in range(100):
	(weights, eOut, iterations) = regression(N, n)
	eOutCounter += eOut
	iterationsCounter += iterations

print "AVG EOUT: ", eOutCounter / 100
print "AVG ITERATIONS: ", iterationsCounter / 100


"""
results = 0.0
outResults = 0.0
for i in range(1000):
	(result, outCounter) = regression(N)
	results += result
	outResults += outCounter

print "FINAL Ein"
print results / 1000
print "FINAL Eout"
print outResults / 1000
"""

