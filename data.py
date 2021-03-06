# -*- coding: utf-8 -*-

# DATA.PY
# Loads dataset from text files.
# Author: Mattia Venturini
#
# Based on Sean Robertson's work:
# https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification

import torch
from torch.autograd import Variable
import glob
import unicodedata
import string
import numpy
import os.path
from io import open
from unidecode import unidecode_expect_nonascii		# per sostituire caratteri non-ASCII

# variabili globali
#category_lines = {}
all_categories = []
n_categories = 0
n_instances = 0

data_X = []
data_Y = []
testX = []
testY = []
validationX = []
validationY = []
index = 0

all_letters = string.ascii_letters + " .,;'-0123456789"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	#s = s.decode('utf-8')		# da stringa a utf-8
	if type(s) == str:
		s = unicode(s, 'utf-8')	# da stringa a utf-8
	s = unidecode_expect_nonascii(s)	# da utf-8 sostituisce caratteri "strani" in stringa ASCII
	s = unicode(s)		# ad unicode

	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# legge da un dataset dove ogni categoria si trova in un file (1 entry per riga)
def dataFromFiles(target='TrainData/*.train.utf8', getData=True, getTestSet=True, getValidationSet=True):

	global all_categories, n_categories, n_instances
	global data_X, data_Y, testX, testY, validationX, validationY

	# azzera il dataset
	all_categories = []
	data_X = []
	data_Y = []
	testX = []
	testY = []
	validationX = []
	validationY = []

	# ricerca nei file di train set
	for index, filename in enumerate(findFiles(target)):
		dirname = os.path.dirname(filename)	# path della cartella
		category = os.path.basename(filename).split('.')[0]	# ricava la categoria dal nome del file
		all_categories.append(category)

		# ricava i relativi dati (se richiesto)
		if getData:
			lines = readLines(filename)
			#category_lines[category] = lines
			data_X += lines
			data_Y += [index for i in lines]

		if getTestSet and os.path.isfile(dirname+"/"+category+".test.utf8"):
			# recupera test set della categoria
			lines = readLines(dirname+"/"+category+".test.utf8")
			testX += lines
			testY += [index for i in lines]

		if getValidationSet and os.path.isfile(dirname+"/"+category+".test.utf8"):
			# recupera validation set della categoria
			lines = readLines(dirname+"/"+category+".validation.utf8")
			validationX += lines
			validationY += [index for i in lines]

	n_instances = len(data_X)
	n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(lines):

	lines = [unicodeToAscii(line) for line in lines]	# converto tutto in ASCII
	lens = [len(line) for line in lines]
	tensor = torch.zeros(max(lens), len(lines), n_letters)
	# The 1st axis is the sequence itself, the 2nd indexes instances in the mini-batch, the 3rd indexes elements of the character.

	for i, line in enumerate(lines):
		for j, letter in enumerate(line):
			tensor[j][i][letterToIndex(letter)] = 1
	return tensor


# rimescola i dati
def shuffle():
	global index

	# mescola X e Y allo stesso modo (serve quindi lo stesso seme)
	seed = 666
	numpy.random.seed(seed)
	numpy.random.shuffle(data_X)
	numpy.random.seed(seed)
	numpy.random.shuffle(data_Y)
	index = 0


# restituisce un'insieme di dati con le rispettive classi
def getBatch(batch_size):

	global index, data_X, data_Y

	index2 = min(index+batch_size, n_instances)

	if index >= index2:		# finite le iterazioni sul DS
		# ricomincia
		shuffle()
		index2 = min(index+batch_size, n_instances)

	# batch
	batch_X = data_X[index : index2]
	batch_Y = data_Y[index : index2]

	index = index2	# aggiorna indice per il batch successivo

	return batch_X, batch_Y, Variable(lineToTensor(batch_X)), Variable(torch.LongTensor(batch_Y))
