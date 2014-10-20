import random
import math
#from numpy import *
import numpy as np
#from numpy.linalg import *
import numpy.linalg as linalg
#from numpy.matlib import *
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

# returns sign(x1^2 + x2^2 - 0.6)
def target(x1, x2):
	result = x1 * x1 + x2 * x2 - 0.6
	return sign(result)




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
	xvalues = range(N)
	yvalues = range(N)

	points = range(N)
	for i in range(N):
		(x,y) = generatePoint()
		points[i] = ((1, x,y, x * y, x * x, y * y))
		xvalues[i] = x
		yvalues[i] = y

	results = range(N)
	for j in range(N):
		point = points[j]
		x = point[1]
		y = point[2]

		res = target(x, y)
		results[j] = res

	# flip sign of random 10%
	for i in range(N/10):
		rand = random.randrange(0,N)
		results[rand] *= -1

	Xa = []
	Ya = []

	Xb = []
	Yb = []
	for i in range(N):
		point = points[i]
		x = point[1]
		y = point[2]
		res = results[i]
		if res == 1:
			Xa.append(x)
			Ya.append(y)
		else:
			Xb.append(x)
			Yb.append(y)

	"""
	t = np.arange(-1, 1.1, 0.1)
	t1 = np.arange(-1, 1.1, 0.1)
	plt.plot(Xa, Ya, 'bs-', Xb, Yb, 'gs-')#, xvalues, yvalues, ':rs')
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""

	# start weights at all 0
	#weights = [0,0,0]
	#print points
	X = np.array(points)
	#print "X: ", X
	#print "X"
	#print X
	XTrans = X.transpose()
	#print "X TRANPOSE"
	#print XTrans

	xTransTimesX = np.dot(XTrans, X)
	#xTransTimesX = xTrans * xiiii
	#print "XT x X"
	#print xTransTimesX
	inverse = linalg.inv(xTransTimesX)

	#print "Inverse"
	#print inverse

	Y = np.array(results)

	w = np.dot(np.dot(inverse, XTrans), Y)
	print w


	#hypSlope = -1 * weights[1] / weights[2]
	#hypIntercept = -1 * weights[0] / weights[2]
	#s = np.arange(-1, 1.5, 0.5)

	"""
	print w
	print weights
	t = np.arange(-1, 1.5, 0.5)
	plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'gs-', xvalues, yvalues, ':rs')
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""
	

	# see how many missclassified points we have
	Ein = np.dot(X,w)
	for i in range(N):
		Ein[i] = sign(Ein[i])
	#Ein -= Y

	#print "Ein"
	#print Ein
	count = 0.0
	for i in range(N):
		#squares += (asscalar(i)^2.0)s
		#print asscalar(Ein[i]),  "   ", results[i]
		if np.asscalar(Ein[i]) != results[i]:
			count += 1

	result = count / N
	print "RESULT: ", result
	return result, w[0], w[1], w[2], w[3], w[4], w[5]

	"""
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
	"""
	t = arange(-1, 1.5, 0.5)
	plt.plot(t, b + slope * t, 'bs-', s, hypIntercept + hypSlope * s, 'gs-', Xa, Ya, ':rs', Xb, Yb, ":bs")
	plt.axis( [-1, 1, -1, 1])
	plt.show()
	"""

	"""
	Xc = []
	Yc = []
	Xd = []
	Yd = []
	EOut = dot(OutArray, w)
	#print EOut
	#print EOut.size
	for i in range(N):
		EOut[i] = sign(EOut[i])
		if EOut[i] != outExpected[i]:
			outCounter += 1
	#print outCounter
	outResult = outCounter / 1000

	return (result, outResult)
	"""

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

N = 500


counter = 0.0
w0count = 0.0
w1count = 0.0
w2count = 0.0
w3count = 0.0
w4count = 0.0
w5count = 0.0
aMissed = 0.0
bMissed = 0.0
cMissed = 0.0
dMissed = 0.0
eMissed = 0.0
for i in range(1000):
	(count, w0, w1, w2, w3, w4, w5) = regression(N)
	# figure out which of the 5 answer choices agrees most given our weights
	
	for j in range(1000):
		(x1, x2) = generatePoint()
		true = sign(1 * w0 + x1 * w1 + x2 * w2 + x1 * x2 * w3 + x1 * x1 * w4 + x2 * x2 * w4 )
		a = sign(-1 - 0.05*x1 + 0.08 * x2 + 0.13 * x1 * x2 + 1.5 * x1 * x1 + 1.5 * x2 * x2 )
		b = sign(-1 - 0.05*x1 + 0.08 * x2 + 0.13 * x1 * x2 + 1.5 * x1 * x1 + 15 * x2 * x2 )
		c = sign(-1 - 0.05*x1 + 0.08 * x2 + 0.13 * x1 * x2 + 15 * x1 * x1 + 1.5 * x2 * x2 )
		d = sign(-1 - 1.5*x1 + 0.08 * x2 + 0.13 * x1 * x2 + .05 * x1 * x1 + .05 * x2 * x2 )
		e = sign(-1 - 0.05*x1 + 0.08 * x2 + 1.5 * x1 * x2 + .15 * x1 * x1 + .15 * x2 * x2 )
		if true != a:
			aMissed += 1
		if true != b:
			bMissed += 1
		if true != c:
			cMissed += 1
		if true != d:
			dMissed += 1
		if true != e:
			eMissed += 1
	counter += count
	w0count += w0
	w1count += w1
	w2count += w2
	w3count += w3
	w4count += w4
	w5count += w5


print "FINAL"
print counter/1000
print w0count/1000
print w1count/1000
print w2count/1000
print w3count/1000
print w4count/1000
print w5count/1000

print "MISSED:"
print aMissed
print bMissed
print cMissed
print dMissed
print eMissed

