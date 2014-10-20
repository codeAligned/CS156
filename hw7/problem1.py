import random
import math
import numpy as np
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

# don't use regularization!
def linearRegression(points, results):
	X = array(points)
	#print X
	XTrans = X.transpose()
	#print XTrans
	xTransTimesX = np.dot(XTrans, X)
	#print xTransTimesX

	#print xTransTimesX.shape[0]
	#k = 1 / float(len(points))
	k = .01
	#print k
	#print k * np.identity(xTransTimesX.shape[0])
	Z = xTransTimesX# + k * np.identity(xTransTimesX.shape[0])

	inverse = inv(Z)
	#print inverse
	Y = array(results)
	#print Y
	w = np.dot(np.dot(inverse, XTrans), Y)
	#print w
	return w

def calcEin(points, results, weights):
	assert len(points) == len(results)
	#print points

	counter = 0
	for i in range(len(points)):
		point = points[i]
		result = results[i]

		sumWeights = 0.0
		for j in range(len(weights)):
			#print point
			#print weights
			sumWeights += (weights[j] * point[j])
		#print sumWeights
		if sign(sumWeights) != result:
			#print "MISS"
			counter += 1
	#print "Count: ", counter
	#print "POINTS: ", len(points)
	return counter / float(len(points))
		

def regression(N):
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
	#xvalues = range(N)
	#yvalues = range(N)

	points = range(N)
	for i in range(N):
		(x,y) = generatePoint()
		points[i] = ((1, x,y))
		#xvalues[i] = x
		#yvalues[i] = y

	results = range(N)
	for j in range(N):
		point = points[j]
		x = point[1]
		y = point[2]

		yVal = slope * x + b
		if (y > yVal):
			results[j] = 1
		else:
			results[j] = -1

	xPos = range(N)


	# start weights at all 0
	#weights = [0,0,0]
	X = array(points)
	#print "X"
	#print X
	XTrans = X.transpose()
	#print "X TRANPOSE"
	#print XTrans

	xTransTimesX = np.dot(XTrans, X)
	#xTransTimesX = xTrans * xiiii
	#print "XT x X"
	#print xTransTimesX

	inverse = inv(xTransTimesX)

	#print "Inverse"
	#print inverse

	Y = array(results)

	w = dot(dot(inverse, XTrans), Y)

	hypSlope = -1 * w[1] / w[2]
	hypIntercept = -1 * w[0] / w[2]
	s = arange(-1, 1.5, 0.5)

	"""
	t = arange(-1, 1.5, 0.5)
	plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'gs-', xvalues, yvalues, ':rs')
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""

	# see how many missclassified points we have
	Ein = dot(X,w)
	for i in range(N):
		Ein[i] = sign(Ein[i])
	#Ein -= Y

	#print "Ein"
	#print Ein
	count = 0.0
	for i in range(N):
		#squares += (asscalar(i)^2.0)s
		#print asscalar(Ein[i]),  "   ", results[i]
		if asscalar(Ein[i]) != results[i]:
			count += 1

	result = count / N

	outCounter = 0.0
	outPoints = range(1000)
	outExpected = range(1000)
	for i in range(1000):
		(x,y) = generatePoint()
		outPoints[i] = (1,x,y)
		yVal = slope * x + b
		if (y > yVal):
			outExpected[i] = 1
		else:
			outExpected[i] = -1
	OutArray = array(outPoints)

	Xa = []
	Ya = []
	Xb = []
	Yb = []
	for i in range(1000): 
		x = outPoints[i][1]
		y = outPoints[i][2]
		res = outExpected[i]
		if res == 1:
			Xa.append(x)
			Ya.append(y)
		else:
			Xb.append(x)
			Yb.append(y)
		#Xa[i] = x
		#Ya[i] = y
	"""
	t = arange(-1, 1.5, 0.5)
	plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'gs-', Xa, Ya, ':rs', Xb, Yb, ":bs")
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""
	

	Xc = []
	Yc = []
	Xd = []
	Yd = []
	EOut = dot(OutArray, w)
	#print EOut
	#print EOut.size
	for i in range(1000):
		EOut[i] = sign(EOut[i])
		if EOut[i] != outExpected[i]:
			outCounter += 1
	#print outCounter
	outResult = outCounter / 1000

	return (result, outResult)

	"""
	count = 0
	converge = 0
	#print "LENG: ", len(points)
	while True:
		# If we have no more misclassified points, we are done
		if checkMisclassified(points, weights, results):
			break
		x = random.randrange(0,N)
		point = points[x]
		yn = results[x]
		#print "YN: " , yn
		xn = list(point)
		#print xn, type(xn)
		#print weights, type(weights)
		result = dotProduct(weights, xn)
		#print result
		#print sign(result)
		if (sign(result) != yn):
			count += 1
			converge = 0
			if (yn == 1):
				weights[0] += xn[0]
				weights[1] += xn[1]
				weights[2] += xn[2]
			else:
				weights[0] -= xn[0]
				weights[1] -= xn[1]
				weights[2] -= xn[2]
		converge += 1

	extraPoints = []
	for i in range(10000):
		(x,y) = generatePoint();
		extraPoints.append((1,x,y))

	#extraResults = []

	miss = 0
	for point in extraPoints:
		x = point[1]
		y = point[2]

		yVal = slope * x + b
		if (y >= yVal):
			answer = 1
			#extraResults.append(1)
		else:
			answer = -1
			#extraResults.append(-1)
		result = dotProduct(list(point), weights)
		if (sign(result) != answer):
			miss += 1

	#print "MISS: ", miss



	return (count, miss)
	"""

N = 100

#regression(N)
def calculateA():
	points = []
	results = []
	validationPoints = []
	validationResults = []
	inputData = open('in.dta', 'r')
	counter = 0
	for line in inputData:
		#print line
		counter += 1
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		if counter <= 25:
			points.append((1, x1, x2, x1 * x1))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			results.append(float(words[2]))
		else:
			validationPoints.append((1, x1, x2, x1 * x1))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			validationResults.append(float(words[2]))

	inputData.close()

	#print points
	#print results
	weights = linearRegression(points, results)
	print "A: use k = 3"
	print "W: ", weights

	#print validationPoints	
	#print validationResults
	eIn = calcEin(validationPoints, validationResults, weights)
	print "Ein: ", eIn

	outPoints = []
	outResults = []
	outData = open('out.dta', 'r')
	for line in outData:
		#print line
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		outPoints.append((1, x1, x2, x1 * x1))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
		outResults.append(float(words[2]))
	outData.close()

	eOut = calcEin(outPoints, outResults, weights)
	print "Eout: ", eOut

def calculateB():
	points = []
	results = []
	validationPoints = []
	validationResults = []
	inputData = open('in.dta', 'r')
	counter = 0
	for line in inputData:
		#print line
		counter += 1
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		if counter <= 25:
			points.append((1, x1, x2, x1 * x1, x2 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			results.append(float(words[2]))
		else:
			validationPoints.append((1, x1, x2, x1 * x1, x2 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			validationResults.append(float(words[2]))

	inputData.close()

	#print points
	#print results
	weights = linearRegression(points, results)
	print "B: use k = 4"
	print "W: ", weights

	#print validationPoints	
	#print validationResults
	eIn = calcEin(validationPoints, validationResults, weights)
	print "Ein: ", eIn

	outPoints = []
	outResults = []
	outData = open('out.dta', 'r')
	for line in outData:
		#print line
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		outPoints.append((1, x1, x2, x1 * x1, x2 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
		outResults.append(float(words[2]))
	outData.close()

	eOut = calcEin(outPoints, outResults, weights)
	print "Eout: ", eOut

def calculateC():
	points = []
	results = []
	validationPoints = []
	validationResults = []
	inputData = open('in.dta', 'r')
	counter = 0
	for line in inputData:
		#print line
		counter += 1
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		if counter <= 25:
			points.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			results.append(float(words[2]))
		else:
			validationPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			validationResults.append(float(words[2]))

	inputData.close()

	#print points
	#print results
	weights = linearRegression(points, results)
	print "C: use k = 5"
	print "W: ", weights

	#print validationPoints	
	#print validationResults
	eIn = calcEin(validationPoints, validationResults, weights)
	print "Ein: ", eIn

	outPoints = []
	outResults = []
	outData = open('out.dta', 'r')
	for line in outData:
		#print line
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		outPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
		outResults.append(float(words[2]))
	outData.close()

	eOut = calcEin(outPoints, outResults, weights)
	print "Eout: ", eOut

def calculateD():
	points = []
	results = []
	validationPoints = []
	validationResults = []
	inputData = open('in.dta', 'r')
	counter = 0
	for line in inputData:
		#print line
		counter += 1
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		if counter <= 25:
			points.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			results.append(float(words[2]))
		else:
			validationPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			validationResults.append(float(words[2]))

	inputData.close()

	#print points
	#print results
	weights = linearRegression(points, results)
	print "D: use k = 6"
	print "W: ", weights

	#print validationPoints	
	#print validationResults
	eIn = calcEin(validationPoints, validationResults, weights)
	print "Ein: ", eIn

	outPoints = []
	outResults = []
	outData = open('out.dta', 'r')
	for line in outData:
		#print line
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		outPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
		outResults.append(float(words[2]))
	outData.close()

	eOut = calcEin(outPoints, outResults, weights)
	print "Eout: ", eOut

def calculateE():
	points = []
	results = []
	validationPoints = []
	validationResults = []
	inputData = open('in.dta', 'r')
	counter = 0
	for line in inputData:
		#print line
		counter += 1
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		if counter <= 25:
			points.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			results.append(float(words[2]))
		else:
			validationPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
			validationResults.append(float(words[2]))

	inputData.close()

	print len(points)
	print len(results)
	print len(validationPoints)
	print len(validationResults)

	#print points
	#print results
	weights = linearRegression(points, results)
	print "E: use k = 7"
	print "W: ", weights

	#print validationPoints	
	#print validationResults
	eIn = calcEin(validationPoints, validationResults, weights)
	print "Ein: ", eIn

	outPoints = []
	outResults = []
	outData = open('out.dta', 'r')
	for line in outData:
		#print line
		words = line.split()
		x1 = float(words[0])
		x2 = float(words[1])
		outPoints.append((1, x1, x2, x1 * x1, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))#, x2 * x2, x1 * x2, abs(x1 - x2), abs(x1 + x2)))
		outResults.append(float(words[2]))
	outData.close()

	eOut = calcEin(outPoints, outResults, weights)
	print "Eout: ", eOut


calculateA()
calculateB()
calculateC()
calculateD()
calculateE()

#print outPoints
#print outResults
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



"""
iterations = 0.0
miss = 0.0
for i in range(1000):
	numIterations, misses = PLA()
	iterations += numIterations
	miss += misses
	print i

avg = iterations/1000
avgMiss = miss / 1000
print "AVG:", avg
print "AVG MISS: ", avgMiss
"""