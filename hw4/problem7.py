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
	return a

def sineA():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)

	# calculate the best b for h(x) = b
	# this will just be the average of y1 and y2 to minimize error
	b = (y1 + y2) / 2
	return b

def sineB():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)

	# calculate the best a for h(x) = ax
	# minimize (a * x1 - y1)^2 + (a * x2 - y2)^2
	# Calculate the best a
	a = ((x1 * y1) + (x2 * y2)) / ((x1*x1) + (x2 * x2))
	return a

def sineC():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)

	# calculate the best a, b for h(x) = ax + b
	# or in other words, just calculate the line between the 2 points... :)
	rise = y2 - y1
	run = x2 - x1
	slope = rise / run
	# find the y intercept
	# y1 = mx1 + b
	# b = y1 - mx1
	b = (y1 - slope * x1)
	return (slope, b)

def sineD():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)

	# calculate the best a for h(x) = ax^2
	# minimize (ax1^2 - y1) ^2 + (ax2^2 - y2) ^2
	# differentiate:
	# 0 = 2(ax1^2 - y1) * x1^2 + 2(ax2^2 - y2) * x2^2
	# ax1^4 - x1^2*y1 + ax2^4 - x2^2 * y2 = 0
	# a(x1^4 + x2^4) = x1^2 * y1 + x2^2 * y2
	# a = (x1^2 * y1 + x2^2 * y2) / (x1 ^ 4 + x2^4)

	num = (x1 * x1 * y1) + (x2 * x2 * y2)
	den = (x1 * x1 * x1 * x1) + (x2 * x2 * x2 * x2)
	a = num/den
	return a

def sineE():
	(x1,y1) = generatePoint()
	y1 = math.sin(math.pi * x1)
	(x2,y2) = generatePoint()
	y2 = math.sin(math.pi * x2)


	# find ax^2 + b == y
	# ax1^2 + b = y1 => b = y1 - ax1^2
	# ax2^2 + b = y2
	# ax2^2 + y1 - ax1^2 = y2
	# a(x2^2 - x1^2) = y2 - y1
	# a = (y2-y1) / (x2^2 - x1^2)

	a = (y2 - y1) + (x2 * x2 - x1 * x1)
	b = y1 - (a * x1 * x1)
	return (a,b)


aCounter = 0
aVariance = 0
for i in range(1000):
	b = sineA()
	aCounter += b
	x = random.uniform(-1,1)
	sq = (b - 0) * (b - 0)
	aVariance += sq
print "A:  AVG B: ", aCounter / 1000
print "A:  AVG VAR: ", aVariance / 1000

aError = 0
for i in range(1000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	g1 = 0 * x1   # g(x) estimate
	aError += ((g1 - y1) * (g1 - y1))
print "A:  AVG BIAS: ", aError / 1000


# CHOICE B
bCounter = 0
bVariance = 0
for i in range(1000):
	a = sineB()
	bCounter += a
	for j in range(1000):
		x = random.uniform(-1,1)
		sq = (a * x - 1.426 * x) * (a * x - 1.426 * x)
		bVariance += sq
print "B:  AVG A: ", bCounter / 1000
print "B:  AVG VAR: ", bVariance / 1000000

bError = 0
for i in range(1000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	g1 = 1.426 * x1   # g(x) estimate
	bError += ((g1 - y1) * (g1 - y1))
print "B:  AVG BIAS: ", bError / 1000

# CHOICE C
cCounterSlope = 0
cCounterIntercept = 0
cVariance = 0
for i in range(1000):
	(slope, b) = sineC()  # g(x)
	cCounterSlope += slope
	cCounterIntercept += b
	#for j in range(1000):
	#	x = random.uniform(-1,1)
	#	sq = (a * x - 1.426 * x) * (a * x - 1.426 * x)
	#	cVariance += sq
print "C:  AVG SLOPE: ", cCounterSlope / 1000
print "C:  AVG INTERCEPT: ", cCounterIntercept / 1000
aAvg = cCounterSlope/1000
bAvg = cCounterIntercept / 1000
for i in range(1000):
	(slope, b) = sineC()
	for j in range(1000):
		x = random.uniform(-1,1)
		sq = (slope * x + b - (aAvg * x + bAvg)) * (slope * x + b - (aAvg * x + bAvg))
		cVariance += sq
print "C:  AVG VAR: ", cVariance / 1000000


cError = 0
for i in range(1000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	#g1 = 1.426 * x1   # g(x) estimate
	g1 = aAvg * x1 + b
	cError += ((g1 - y1) * (g1 - y1))
print "C:  AVG BIAS: ", cError / 1000

# CHOICE D
dCounterSlope = 0
dVariance = 0
for i in range(10000):
	slope = sineD()
	dCounterSlope += slope
print "D:  AVG SLOPE: ", dCounterSlope / 10000
dAvg = dCounterSlope / 10000


for i in range(1000):
	slope = sineD()
	for j in range(1000):
		x = random.uniform(-1,1)
		sq = (slope * x * x - (dAvg * x * x)) * (slope * x * x - (dAvg * x * x))
		dVariance += sq
print "D:  AVG VAR: ", dVariance / 1000000


dError = 0
for i in range(1000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	g1 = dAvg * x1 * x1  # g(x) estimate
	dError += ((g1 - y1) * (g1 - y1))
print "D:  AVG BIAS: ", dError / 1000

# CHOICE E
eCounterSlope = 0
eCounterIntercept = 0
eVariance = 0
for i in range(1000):
	(a,b) = sineE()
	eCounterSlope += a
	eCounterIntercept += b
print "E:  AVG SLOPE: ", eCounterSlope / 1000
print "E:  AVG INTERCEPT: ", eCounterIntercept / 1000
eAvg = eCounterSlope / 1000
eAvgInt = eCounterIntercept/1000


for i in range(1000):
	(a, b) = sineE()
	for j in range(1000):
		x = random.uniform(-1,1)
		sq = (a * x * x + b - (eAvg * x * x + eAvgInt)) * (a * x * x + b - (eAvg * x * x + eAvgInt))
		eVariance += sq
print "E:  AVG VAR: ", eVariance / 1000000


eError = 0
for i in range(1000):
	(x1, y1) = generatePoint()   # True f(x1) = y1
	y1 = math.sin(math.pi * x1)

	g1 = eAvg * x1 * x1 + eAvgInt  # g(x) estimate
	eError += ((g1 - y1) * (g1 - y1))
print "E:  AVG BIAS: ", eError / 1000




"""
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
"""

