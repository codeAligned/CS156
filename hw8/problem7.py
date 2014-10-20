import commands
import random
import time

counter0 = 0
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0

total0 = 0
total1 = 0
total2 = 0
total3 = 0
total4 = 0
for i in range(100):
	time.sleep(1)
	output1 = commands.getstatusoutput('./libsvm-3.17/svm-train -t 1 -d 2 -g 1 -r 1 -v 10 -c 0.0001 1vs5.train 1vs5.train.001.CV.model')
	acc1 = float(output1[1][-8:-1])
	print acc1

	output2 = commands.getstatusoutput('./libsvm-3.17/svm-train -t 1 -d 2 -g 1 -r 1 -v 10 -c 0.001 1vs5.train 1vs5.train.001.CV.model')
	acc2 = float(output2[1][-8:-1])
	print acc2

	output3 = commands.getstatusoutput('./libsvm-3.17/svm-train -t 1 -d 2 -g 1 -r 1 -v 10 -c 0.01 1vs5.train 1vs5.train.001.CV.model')
	acc3 = float(output3[1][-8:-1])
	print acc3

	output4 = commands.getstatusoutput('./libsvm-3.17/svm-train -t 1 -d 2 -g 1 -r 1 -v 10 -c 0.1 1vs5.train 1vs5.train.001.CV.model')
	acc4 = float(output4[1][-8:-1])
	print acc4

	output5 = commands.getstatusoutput('./libsvm-3.17/svm-train -t 1 -d 2 -g 1 -r 1 -v 10 -c 1 1vs5.train 1vs5.train.001.CV.model')
	acc5 = float(output5[1][-8:-1])
	print acc5

	minIndex = -1
	minimumError = 0
	for index, accuracy in enumerate([acc1, acc2, acc3, acc4, acc5]):
		#print index, error
		if accuracy > minimumError:  
			minimumError = accuracy
			minIndex = index

	print "minIndex: ", minIndex
	print "minError: ", minimumError

	total0 += acc1
	total1 += acc2
	total2 += acc3
	total3 += acc4
	total4 += acc5

	if minIndex == 0:
		counter0 += 1
	elif minIndex == 1:
		counter1 += 1
	elif minIndex == 2:
		counter2 += 1
	elif minIndex == 3:
		counter3 += 1
	elif minIndex == 4:
		counter4 += 1
	print ""

print "C = 0.0001 wins: ", counter0
print "C = 0.001 wins: ", counter1
print "C = 0.01 wins: ", counter2
print "C = 0.1 wins: ", counter3
print "C = 1.0 wins: ", counter4

print "C = 0.0001 avg: ", total0 / 100.0
print "C = 0.001 avg: ", total1 / 100.0
print "C = 0.01 avg: ", total2 / 100.0
print "C = 0.1 avg: ", total3 / 100.0
print "C = 1.0 avg: ", total4 / 100.0



