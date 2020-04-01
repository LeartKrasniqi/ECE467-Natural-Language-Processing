# Leart Krasniqi
# ECE467: Natural Language Processing
# Prof. Sable
# Spring 2020

# This script is an implementation of the CKY Algorithm to parse sentences according to a grammar.
# It assumes the grammar is given in Chomsky Normal Form (CNF).
# Influenced by implementation on Wikipedia, which is only slightly different from the one presented in class: https://en.wikipedia.org/wiki/CYK_algorithm


# Node class to be used for storing data for a non-terminal symbol
# self.symbol -> Stores the symbol itself
# self.r1 -> Stores the first member on the RHS (i.e. a terminal or another non-terminal)
# self.r2 -> Stores the second member on the RHS (i.e. a non-terminal)
class Node:
	def __init__(self, symbol, r1, r2=None):
        self.symbol = symbol
        self.r1 = r1
        self.r2 = r2

def gen_tree():




def gen_tree_text(node):
	if node.r2 == None:
		string = "[" + node.symbol + " " + node.r1 + "]"
		return string
	else
		string = "[" + node.symbol + " " + gen_tree_text(node.r1) + " " + gen_tree_text(node.r2) + "]"
		return string

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
	line.split()
	temp_list = []

	# Append the symbols, ignoring the '-->' and accounting for 1 or 2 right-hand-side symbols 
	temp_list.append(line[0])
	temp_list.append(line[2])
	if(len(temp_list) == 4):
		temp_list.append(line[3])

	grammar_rules.append(temp_list)


# Run until user types 'quit'
while(True):

	# Read input from user
	sentence = input("Type in your sentence (or type quit to exit): ")
	tokens = sentence.split()

	# If 'quit', exit
	if(tokens[0] == "quit"):
		print("Session Ended")
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
		for cell in range(0, (length - l) + 1):
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
							if n.symbol == rule[2]:
								r_nodes.append(n)

						# Create nodes which contain l_nodes and r_nodes as children, and append to cell in table
						for ln in l_nodes:
							for rn in r_nodes:
								table[words - 1][cell].append( Node(rule[0], ln, rn) )


















