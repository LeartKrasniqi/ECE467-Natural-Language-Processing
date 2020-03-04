# # Leart Krasniqi
# ECE467: Natural Language Processing
# Prof. Sable
# Spring 2020

# This script is used to classify text using the Naive-Bayes method.
# The user is prompted for a file containing the train statistics and a file containing the unlabeled documents

import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from math import log

# Cool progress bar
from progress.bar import IncrementalBar

# Obtain relevant files from user
#test_filename = input("Please provide the name of the file containing the unlabeled documents: ")
#stats_filename = input("Please provide the name of the file containing the training statistics: ")

# Hardcoded for testing purposes 
test_filename = "corpus1_test.list"
stats_filename = "stats1.json"
print("\n")

# Convert the statistics JSON into a dictionary 
json_str = open(stats_filename).read()
stats = json.loads(json_str)

# All of the categories
categories = stats.keys()
	
# Get total number of training files
num_training_files = 0
for c in categories:
	num_training_files += stats[c]["num_files"]

# Get each filename 
test_file = open(test_filename, "r")
test_lines = test_file.read().splitlines()

# Stemmer
stemmer = PorterStemmer()

# List to hold the predictions
output_list = []

bar = IncrementalBar("Classifying:", max = len(test_lines))
# Check each unlabeled file
for line in test_lines:
	f_test = open(line, "r") 

	# Tokenize
	tokens = word_tokenize(f_test.read())

	# Construct a dictionary containing tokens and their frequencies
	token_count = dict()
	for token in tokens:
		t = stemmer.stem(token)

		# Skip punctuation
		if t in string.punctuation:
			pass
		else:
			if t not in token_count.keys():
				token_count[t] = 1
			else:
				token_count[t] += 1

	# Get the vocabulary size by finding the number of unique tokens
	vocab_size = len(token_count)
	
	# Go through each category and find P(category|document), using the Naive assumption
	# Store the log probs in a dict
	category_log_probs = dict()
	for c in categories:
		# P(c) is based on frequency of category in the training set 
		log_prior_prob = log(stats[c]["num_files"] / num_training_files)

		# Laplace Smoothing Variable
		k = 1

		# Calculate counts of words in the specific category
		log_category_prob = 0
		for t in token_count.keys():
			if t in stats[c]["tokens"].keys():
				c_w = stats[c]["tokens"][t]
			else:
				c_w = 0

			# Log prob with smoothing
			log_token_category_prob = log( (c_w + k) / (stats[c]["num_tokens"] + k*vocab_size) ) * token_count[t]
			log_category_prob += log_token_category_prob

		# The total log prob for the category is the sum of the prior prob and the sums of all the token probs
		category_log_probs[c] = log_prior_prob + log_category_prob
		
	# Use argmax to make decision for the category
	winner = max(category_log_probs, key = category_log_probs.get)

	# Append to list of outputs
	output_list.append(line + " " + winner + "\n")

	bar.next()

bar.finish()
print("\nDone classifying!\n")
	
# Prompt user for name of file to output results
out_filename = input("Please provide a name for the prediction output file: ")

# Write results to outfile
outfile = open(out_filename, "w")
for decision in output_list:
	outfile.write(decision)

outfile.close()















