# Script to split training data into training/test sets

import sys, os, string

if len(sys.argv) != 5:
	print("Usage: python3 train_test_split.py [training_file] [new_train_file] [new_test_file] [training_percentage]")
	exit()

original_training_file = open(sys.argv[1], "r")
original_lines = original_training_file.read().splitlines()
percentage = int(sys.argv[4])/100

new_train = []
new_test = []

# Dictionary to hold category counts
cat_counts = dict()
for line in original_lines:
	split = line.split()
	cat = split[1]

	if cat not in cat_counts.keys():
		cat_counts[cat] = 1
	else:
		cat_counts[cat] += 1

# Get only (percentage)% of each category
for cat in cat_counts.keys():
	cat_counts[cat] = int(percentage * cat_counts[cat])

# Go through each file and only train on (percentage)% of each category
for line in original_lines:
	split = line.split()
	file = split[0]
	cat = split[1]

	if cat_counts[cat] > 0:
		new_train.append(line + "\n")
		cat_counts[cat] -= 1
	else:
		new_test.append(file + "\n")

new_train_file = open(sys.argv[2], "w")
new_test_file = open(sys.argv[3], "w")

# Write the new data to the files
for t in new_train:
	new_train_file.write(t)

for t in new_test:
	new_test_file.write(t)

new_train_file.close()
new_test_file.close()
original_training_file.close()

