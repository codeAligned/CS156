import random
import cvxopt
import numpy as np
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

# returns False if we have a missclassifed points
def checkMisclassified(points, weights, results):
	for i in range(len(points)):
		result = sign(dot(weights, list(points[i])))
		if (result != results[i]):
			return False
	return True



def PLA(N):
	(x1,y1) = generatePoint()
	(x2,y2) = generatePoint()

	rise = y2 - y1
	run = x2 - x1

	slope = rise / run

	# y = mx + b
	#b = y - mx
	b = y1 - slope * x1

	# Now, we have y = (slope) * x + b

	points = range(N)
	results = range(N)
	for i in range(N):
		(x,y) = generatePoint()
		points[i] = ((1, x,y))

		yVal = slope * x + b
		if (y >= yVal):
			results[i] = 1
		else:
			results[i] = -1;


	# start weights at all 0
	weights = [0,0,0]

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
		result = dot(weights, xn)
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

	extraPoints = range(1000)
	miss = 0
	for i in range(1000):
		(x,y) = generatePoint();
		extraPoints[i] = ((1,x,y))

		yVal = slope * x + b
		if (y >= yVal):
			answer = 1
			#extraResults.append(1)
		else:
			answer = -1
			#extraResults.append(-1)
		point = [1, x, y]
		result = dot(list(point), weights)
		if (sign(result) != answer):
			miss += 1

	#extraResults = []

	#print "MISS: ", miss



	return (count, miss)

N = 100
iterations = 0.0
miss = 0.0
#numIterations, misses = PLA(N)
#print "It: ", numIterations
#print "misses: ",  misses

for i in range(1000):
	numIterations, misses = PLA(N)
	print "I: ", i
	print "It: ", numIterations
	print "misses: ",  misses
	iterations += numIterations
	miss += misses
	#print i

"""
avg = iterations/1000
avgMiss = miss / 1000
print "AVG:", avg
print "AVG MISS: ", avgMiss
"""






