# Script to run the system a bunch of times and figure out the best value of k
import numpy as np
import smoothing_categorize as sc 
import subprocess

c1 = dict()
c2 = dict()
c3 = dict()
sums = dict()

file = open("k_stats_fine.txt", "w")
file.write("[")
for k in np.linspace(0.001, 0.100, 100):
	sc.categorize("corpus1_test.list", "stats1.json", "1", k)
	sc.categorize("c2_test.list", "stats2.json", "2", k)
	sc.categorize("c3_test.list", "stats3.json", "3", k)

	p1 = subprocess.Popen(["perl", "analyze.pl", "smoothing_out_1.labels", "corpus1_test.labels"], stdout = subprocess.PIPE)
	out1 = str(p1.communicate())
	out1 = out1.split("RATIO = ")
	ratio1 = out1[1][0:10]

	p2 = subprocess.Popen(["perl", "analyze.pl", "smoothing_out_2.labels", "corpus2_train.labels"], stdout = subprocess.PIPE)
	out2 = str(p2.communicate())
	out2 = out2.split("RATIO = ")
	ratio2 = out2[1][0:10]

	p3 = subprocess.Popen(["perl", "analyze.pl", "smoothing_out_3.labels", "corpus3_train.labels"], stdout = subprocess.PIPE)
	out3 = str(p3.communicate())
	out3 = out3.split("RATIO = ")
	ratio3 = out3[1][0:10]

	try:
		c1[k] = float(ratio1)
	except:
		c1[k] = float(ratio1[0:2])
	try:
		c2[k] = float(ratio2)
	except:
		c2[k] = float(ratio2[0:2])
	try:
		c3[k] = float(ratio3)
	except:
		c3[k] = float(ratio3[0:2])
	c1[k] = ratio1
	c2[k] = ratio2
	c3[k] = ratio3
	sums[k] = c1[k] + c2[k] + c3[k]
	file.write(str(k) + ",")

file.write("]\n[")
for k in c1.keys():
	file.write(c1[k] + ",")
file.write("]\n[")
for k in c2.keys():
	file.write(c2[k] + ",")
file.write("]\n[")
for k in c3.keys():
	file.write(c3[k] + ",")
file.write("]")

max1 = max(c1, key = c1.get)
max2 = max(c2, key = c2.get)
max3 = max(c3, key = c3.get)
max_sum = max(sums, key = sums.get)

print("For C1, k = " + str(max1))
print("For C2, k = " + str(max2))
print("For C3, k = " + str(max3))
print("For SUM, k = " + str(max_sum))

	


	 