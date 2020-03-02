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

# Obtain relevant files from user
test_filename = input("Please provide the name of the file containing the unlabeled documents: ")
stats_filename = input("Please provide the name of the file containing the training statistics: ")

# Convert the statistics JSON into a dictionary 
json_str = open(stats_filename).read()
stats = json.loads(json_str)

# Get each filename 
test_file = open(test_filename, "r")
test_lines = test_file.read().splitlines()

# Stemmer
stemmer = PorterStemmer()

# Dictionary to hold counts of tokens
token_count = dict()

# Check each line of the unlabeled file
for line in test_lines:
	f_test = open(line, "r")

	# Tokenize
	tokens = word_tokenize(f_test.read())

	# Construct a dictionary containing tokens and their frequencies
	for token in tokens:
		t = stemmer.stem(token)

		# Skip punctuation
		if t in string.punctuation:
			pass
		else:
			if t not in token_count:
				token_count[t] = 1
			else:
				token_count[t] += 1

	# Get the vocabulary size by finding the number of unique tokens
	vocab_size = len(token_count)

	# TODO: Implement the classifier and write to file



