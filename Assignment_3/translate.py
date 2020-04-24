# Leart Krasniqi
# ECE467: Natural Language Processing
# Prof. Sable
# Spring 2020

# This script uses an RNN Encoder-Decoder Model with TensorFlow to translate Italian text into English  
# The main steps are Preprocessing, Modeling, Predicting, and Iterating

import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import io 
import sys


if len(argv) != 2:
	print("Usage: python3 translate.py [data_file]")
	exit(-1)


#############################
#       Preprocessing       #      
#############################

# Function to convert unicode to ascii
def unicode_to_ascii(sentence):
	ret = "".join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')
	return ret


# Function to normalize sentences into lowercase words split by spaces
# Ex:
# 	"This is a sentence."  ->  "<s> this is a sentence . <e>" 
def preprocess_sentence(sentence):
	s_ascii = unicode_to_ascii(sentence)

	# Create spaces between the words and punctuation, and replace other characters with a space
	norm = re.sub(r"([?.!,])", r" \1 ", s_ascii)
	norm = re.sub(r'[" "]+', " ", norm)
	norm = re.sub(r"[^a-zA-Z?.!,]+", " ", norm)

	# Remove leading and trailing spaces
	norm = norm.split()

	# Append starting and ending markers
	norm = "<s> " + norm + " <e>"
	return norm


# Function to tokenize a list of sentences (with padding)
# Returns both the tokenized data and the tokenizer itself
def tokenize(sentences):
	tokenizer = tf.keras.preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(sentences)

	tensor = tokenizer.texts_to_sequences(sentences)
	tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

	return tensor, tokenizer


# Function to actually create the dataset
# Returns word pairs in format: [ENGLISH, ITALIAN]
def create_dataset(path, num_examples):
	# Read in lines from data file, which are in format:
	# [English Sentence][\t][Italian Sentence]
	lines = io.open(path, encoding='UTF-8').read.strip().split('\n')

	# Construct the word pairs for each line
	pairs = [[preprocess_sentence(s) for s in l.split('\t')]  for l in lines[:num_examples]]

	return zip(*word_pairs)


# Function to load the data set (i.e. obtain tensors from the word pairs)
# Returns the tensors and the tokenizers
def load_dataset(path, num_examples=None):
	# Get the word pairs
	target_lang, input_lang = create_dataset(path, num_examples)

	# Obtain the tensors and the tokenizers
	target_tensor, target_tokenizer = tokenize(target_lang)
	input_tensor, input_tokenizer = tokenize(input_lang)

	return input_tensor, target_tensor, input_tokenizer, target_tokenizer



num_examples = 100000
input_tensor, target_tensor, input_lang, target_lang = load_dataset(argv[1], num_examples)

# Find max length of the tensors
max_len_target, max_len_input = target_tensor.shape[1], input_tensor.shape[1]

# Use 80% for training, 20% for validation
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Create the dataset
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
input_vocab_size = len(input_lang.word_index) + 1
target_vocab_size = len(target_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)	








#########################
#       Modelling       #
#########################

# Encoder model
# Used for the input language (i.e. Italian text)
# Inspired by tensorflow documentation of NMT: https://github.com/tensorflow/nmt
class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
		super(Encoder, self).__init__()
		self.enc_units = enc_units
		self.batch_size = batch_size
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

		# Gated recurrent unit, like LSTM but more efficient
		self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, intial_state=hidden)
		return output, state

	def init_hidden_state(self):
		return tf.zeros((self.batch_size, self.enc_units))

# Attention class 
# Used to help with limitation that encoder must compress necessary info into a fixed-length vector, which
# ultimately causes poor performance when longer sentences are found in training corpus
# Implementation based on: 	https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
#							https://talbaumel.github.io/blog/attention/ 
# Alternatively, can:
#	pip3 install tensorflow-addons 
#	from tensorflow_addon.sseq2seq import BahdanauAttention 
class Attention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(Attention, self).__init__()
		self.w1 = tf.keras.layers.Dense(units)
		self.w2 = tf.keras.layers.Dense(units)
		self.v = tf.keras.layers.Dense(1)

	def call(self, query, values):
		query_with_time_axis = tf.expand_dims(query, 1)
		score = self.v( tf.nn.tanh(self.w1(query_with_time_axis) + self.w2(values)) )
		attention_weights = tf.nn.softmax(score, axis=1)

		context_vec = attention_weights * values
		context_vec = tf.reduce_sum(context_vec, axis=1)

		return context_vec, attention_weights

# Decoder model
# Used for the target language (i.e. English text)
class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
		super(Decoder, self).__init__()
		self.dec_units = dec_units
		self.batch_size = batch_size
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		self.attention = Attention(self.dec_units)

	def call(self, x, hidden, enc_output):
		x = self.embedding(x)
		context_vec, attention_weights = self.attention(hidden, enc_output)

		# Concat embedding with the context vec that comes from attention
		x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)

		# Pass concat vector to GRU
		output, state = self.gru(x)

		output = tf.reshape(output, (-1, output.shape[2]))
		x = self.fc(output)

		return x, state, attention_weights


# Loss function
def loss_fn(predicted, actual, loss_object):
	mask = tf.math.logical_not(tf.math.equal(actual, 0))
	loss = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss.dtype)
	loss += mask

	return tf.reduce_mean(loss)



# Create the encoder and decoder models
encoder = Encoder(input_vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(target_vocab_size, embedding_dim, units, BATCH_SIZE)

# Define loss object and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()

# Saving checkpoints
chkpt_dir = "./tf_ckpts"
chkpt_prefix = os.path.join(chkpt_dir, "chkpt")
chkpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)






########################
#       Training       #     
########################

@tf.function
def train_step(input, target, enc_hidden):
	loss = 0

	with tf.GradientTape() as tape:
		enc_output, enc_hidden = encoder(input, enc_hidden)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims( [target_lang.word_index['<s>']] * BATCH_SIZE, 1)

		# Use teacher forcing, which is a technique where the target word is passed as the next input to decoder
		for t in range(1, target.shape[1]):
			predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
			loss += loss_function(target[:, t], predictions, loss_object)
			dec_input = tf.expand_dims(target[:, t], 1)

	batch_loss = loss / int(target.shape[1])
	var = encoder.trainable_variables + encoder.trainable_variables
	gradients = tape.gradient(loss, variables)

	optimizer,apply_gradients(zip(gradients, variables))

	return batch_loss


# Perform the training
EPOCHS = 10

for epoch in range(EPOCHS):
	start = time.time() 
	
	enc_hidden = encoder.init_hidden_state()
	total_loss = 0

	# Calculate the batch loss for each batch and then append to total loss
	for (batch, (inp, target)) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(inp, target, enc_hidden)
		total_loss += batch_loss

		# Report status for every 100 batches (for my sanity)
		if batch % 100 == 0:
			print("Epoch {} Batch {} Loss {:.3f}".format(epoch+1, batch, batch_loss))

		# Save checkpoints every epoch
		checkpoint.save(file_prefix = chkpt_prefix)

		# Report loss and total time spent
		print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss/steps_per_epoch))
		print('Time taken for Epoch {}: {} sec\n'.format(epoch + 1, time.time() - start))


def main(argv):
	
	

	

	

	
	

if __name__ == "__main__":
	main(sys.argv)










