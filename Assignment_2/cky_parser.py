# Leart Krasniqi
# ECE467: Natural Language Processing
# Prof. Sable
# Spring 2020

# This script is an implementation of the CKY Algorithm to parse sentences according to a grammar.
# It assumes the grammar is given in Chomsky Normal Form (CNF).
# Influenced by implementation on Wikipedia, which is only slightly different from the one presented in class: https://en.wikipedia.org/wiki/CYK_algorithm

import string 
import sys

# Node class to be used for storing data for a non-terminal symbol
# self.symbol -> Stores the symbol itself
# self.r1 -> Stores the first member on the RHS (i.e. a terminal or another non-terminal)
# self.r2 -> Stores the second member on the RHS (i.e. a non-terminal)
class Node:
	def __init__(self, symbol, r1, r2 = None):
		self.symbol = symbol
		self.r1 = r1
		self.r2 = r2

# Helper function to print spaces to make the tree representation look pretty
def print_spaces(level):
	for i in range(0, level):
		sys.stdout.write("  ")

# Function to print the parse more like a tree, rather than a list
def print_tree(node, level):
	print_spaces(level)

	if node.r2 == None:
		sys.stdout.write(node.symbol + "\n")
		print_spaces(level + 1)
		sys.stdout.write(node.r1 + "\n")
	else:
		sys.stdout.write(node.symbol + "\n")
		print_tree(node.r1, level + 1)
		print_tree(node.r2, level + 1)

# Function to create a string of a list that represents the parse
def gen_tree_text(node):
	if node.r2 == None:
		string = "[" + node.symbol + " " + node.r1 + "]"
		return string
	else:
		string = "[" + node.symbol + " " + gen_tree_text(node.r1) + " " + gen_tree_text(node.r2) + "]"
		return string

def main():
	# Prompt user for file containing CNF of grammar
	grammar_filename = input("Please provide the name of the file containing the CNF of the grammar: ")
	grammar_file = open(grammar_filename, "r")
	grammar_lines = grammar_file.read().splitlines()

	# Turn the grammar lines into a list of lists
	# Example:
	#	A --> B C
	#	C --> D
	# This grammar becomes:
	# [ [A, B, C], [C, D] ]
	grammar_rules = []
	for line in grammar_lines:
		line = line.split()
		temp_list = []

		# Append the symbols, ignoring the '-->' and accounting for 1 or 2 right-hand-side symbols 
		temp_list.append(line[0])
		temp_list.append(line[2])
		if(len(line) == 4):
			temp_list.append(line[3])

		grammar_rules.append(temp_list)


	# String manipulation 
	translator = str.maketrans('', '', string.punctuation)

	# Run until user types 'quit'
	while(True):

		# Read input from user
		sentence = input("Type in your sentence (or type 'quit' to exit): ")

		# Remove puncutation and set string to all lowercase (the grammar expects this)
		new_sentence = sentence.translate(translator).lower()

		# Split into tokens
		tokens = new_sentence.split()

		# If 'quit', exit
		if new_sentence == "quit":
			print("Parsing session ended")
			exit(0)

		# Store number of tokens in sentence
		length = len(tokens)

		# Create an empty table which will be used for parsing
		table = [ [ [] for i in range(length - j)]  for j in range(length) ]

		# Put in the nodes containing the terminals in the first row of the table
		word_idx = 0
		for token in tokens:
			for rule in grammar_rules:
				if token == rule[1]:
					table[0][word_idx].append( Node(rule[0], token) )

			word_idx += 1



		# Perform the CKY Algorithm 
		# words = remaining words in the sentence
		# cell = current cell in the table
		# l_part = left partition of sentence
		for words in range(2, length + 1): 
			for cell in range(0, (length - words) + 1):
				for l_part in range(1, words):
					r_part = words - l_part

					l_cell = table[l_part - 1][cell]
					r_cell = table[r_part - 1][cell + l_part]

					# Fill in the left and right cells depending on the grammar rule
					for rule in grammar_rules:
						l_nodes = []
						for n in l_cell:
							if n.symbol == rule[1]:
								l_nodes.append(n)

						# Need the if here to make sure at least one l_node has actually been created
						if l_nodes:
							r_nodes = []
							for n in r_cell:
								if len(rule) == 3:
									if n.symbol == rule[2]:
										r_nodes.append(n)


							# Create nodes which contain l_nodes and r_nodes as children, and append to cell in table
							for ln in l_nodes:
								for rn in r_nodes:
									table[words - 1][cell].append( Node(rule[0], ln, rn) )



		# Print the parse
		# The starting symbol for the grammar will always be 'S'
		start_symbol = "S"

		# Read first column of table from BOTTOM to TOP (hence the -1 start index)
		nodes = []
		for n in table[-1][0]:	
			if n.symbol == start_symbol:
				nodes.append(n)

		# If the nodes list is nonempty, then we have a valid parse
		if nodes:
			it = 1
			for n in nodes:
				print("\nParse #" + str(it) + " (List-Form): ")
				print(gen_tree_text(n) + "\n")

				print("Parse #" + str(it) + " (Tree-Form): ")
				print_tree(n, 0)
				print("\n\n")

				it += 1
		else:
			print("No valid parses!")


if __name__ == "__main__":
    main()




















