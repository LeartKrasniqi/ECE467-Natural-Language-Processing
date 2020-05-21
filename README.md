# ECE467-Natural-Language-Processing
Projects in NLP course

## Assignment 1: Text Classification
Text classification using a Naive Bayes classifier.

#### To Run
```bash
# Train the classifier (Will be prompted for training file and output file)
python3 train.py

# Run the classifier (Will be prompted for testing file, file with training stats, and output file)
python3 categorize.py 

# Check accuracy
perl analyze.pl [predictions] [actual]
```


## Assignment 2: Parsing
Parsing sentences using the CKY Algorithm.

#### To Run 
```bash
# Convert CFG into CNF (if needed)
python3 CFG_to_CNF.py [CFG_file] [CNF_file]

# Run the parser (First prompts user for CNF file and then receives input from stdin and displays valid parses)
python3 cky_parser.py
```

## Assignment 3: Machine Learning
Machine translation using TensorFlow.

#### To Run
```bash
# Can train the model with the provided datafile
python3 translate.py [datafile] train

# Alternatively, download checkpoints from a pre-trained model
wget https://ee.cooper.edu/~krasniqi/NLP/training.zip

# Run the program with the pre-trained model (the translate flag allows for input from stdin)
python3 translate.py [datafile] [test/translate]
```
