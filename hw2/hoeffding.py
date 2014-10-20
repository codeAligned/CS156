import random


def experiment():
	coins = []
	for i in range(1000):
		total = 0.0
		for j in range(10):
			flip = random.randrange(0,2) # random from 0 to 1
			#print flip
			total += flip
		coins.append(total)

	v1 = coins[0] / 10
	minimum = min(coins)
	vmin = minimum / 10
	x = random.randrange(0,1000)  # get a random element
	vrand = coins[x] / 10
	result = (v1, vmin, vrand)
	#print result
	return result


# run the experiment 100000 times
avgV1 = 0.0
avgVMin = 0.0
avgVRand = 0.0
for i in range(100000):
	print i
	(v1, vmin, vrand) = experiment()
	avgV1 += v1
	avgVMin += vmin
	avgVRand += vrand

avgV1 /= 100000
avgVMin /= 100000
avgVRand /= 100000

print "AVG: ", avgV1, "  ", avgVMin,  "  ", avgVRand