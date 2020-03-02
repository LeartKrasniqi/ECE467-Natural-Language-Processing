# Leart Krasniqi
# ECE467: Natural Language Processing
# Prof. Sable
# Spring 2020

# This script is used to train the Naive-Bayes classifier.  
# The user provides an input file containing training data and the name of the desired output file where the training statistics are saved.

import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Obtain file name from user
train_filename = input("Please provide the name of the file containing the training set:")

print("\nCalculating statistics...\n")

train_file = open(train_filename, "r")
train_lines = train_file.read().splitlines()

# Stemmer
stemmer = PorterStemmer()

# Have a dictionary with the stats, which will ultimately be saved as a JSON
# Format:
# 	{
# 		[CATEGORY]:
# 		{
# 			num_tokens: [COUNT],
# 			num_files: [COUNT],
# 			tokens:
# 			{
# 				[TOKEN]: [COUNT],
# 			},
# 		},	
# 	}

d = dict()

# Loop through each line in the file and determine statistics
# Format: 	/path/to/file category
for line in train_lines:
	# Split each line into [/path/to/file, category]
	split = line.split()

	f_train = open(split[0], "r")
	category = split[1]
	
	# Append categories to our stats as we go along
	if category not in d.keys():
		d[category] = {}
		d[category]["num_files"] = 1
		d[category]["num_tokens"] = 0
		d[category]["tokens"] = {}
	else:
		d[category]["num_files"] += 1

	# Tokenize
	tokens = word_tokenize(f_train.read())

	# Now, compute stats for each token 
	for token in tokens:
		# Apply Porter Stemmer
		t = stemmer.stem(token)	

		# Frequency of this token
		if t not in d[category]["tokens"].keys():
			d[category]["tokens"][t] = 1
		else:
			d[category]["tokens"][t] += 1

		# Increment the number of tokens in the category
		d[category]["num_tokens"] += 1


print("Done calculating statistics!\n")

# Prompt user to supply the output file
outfile = input("Please provide the name of the file to save the statsitics (ex: stats.json):")

# Save the dictionary with stats to the JSON file
with open(outfile, 'w') as fp:
    json.dump(d, fp)















