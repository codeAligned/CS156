points = []
results = []
validationPoints = []
validationResults = []
inputData = open('test.dta', 'r')
f = open('0vsAll.test','w')
#counter = 0
for line in inputData:
	#print line
	#counter += 1
	words = line.split()
	x1 = float(words[0])
	x2 = float(words[1])
	x3 = float(words[2])
	if x1 == 0:
		string = '+1 1:' + words[1] + ' 2:' + words[2] + '\n'
	else:
		string = '-1 1:' + words[1] + ' 2:' + words[2] + '\n'
	print words
	f.write(string) # python will convert \n to os.linesep

	
inputData.close()
f.close() # you can omit in most cases as the destructor will call if