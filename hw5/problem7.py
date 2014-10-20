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

def dU(u, v):
	return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))

def dV(u, v):
	return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))

def E(u,v):
	return (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * v * math.exp(-u))

u = 1.0
v = 1.0
n = 0.1
#counter = 0
error = E(u, v)
for i in range(15):
	du = dU(u, v)
	dv = dV(u,v)
	#length = du * du + dv * dv 
	#vx = du/ length
	#vy = dv / length
	u = u - n*du
	du2 = dU(u, v)
	dv2 = dV(u,v)
	v = v - n*dv2
	#counter += 1
	error = E(u, v)
	print du, "  ", dv, "  ", error
	print "UV:", u , "  ", v 
	if error < .00000000000001:
		break
	#print "ER: ", error
#print "COUNTER:  ", counter
print "error: " , error 



