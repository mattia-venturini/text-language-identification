# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable	
import glob
import unicodedata
import string
import numpy
from io import open
from unidecode import unidecode_expect_nonascii		# per sostituire caratteri non-ASCII

# variabili globali
category_lines = {}
all_categories = []
n_categories = 0
n_instances = 0

data_X = []
data_Y = []
index = 0

all_letters = string.ascii_letters + " .,;'-"
#all_letters = string.ascii_lowercase + " .,;'-"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
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
def dataFromFiles(target='TrainData/*.utf8', getData=True):
	
	global category_lines, all_categories, n_categories, n_instances
	global data_X, data_Y
	
	# Build the category_lines dictionary, a list of lines per category
	category_lines = {}
	all_categories = []
	
	for index, filename in enumerate(findFiles(target)):
		# ricava la categoria
		category = filename.split('/')[-1].split('.')[0]
		all_categories.append(category)
		
		# ricava i relativi dati (se richiesto)
		if getData:
			lines = readLines(filename)
			category_lines[category] = lines
			
			data_X += lines
			data_Y += [index for i in lines]
			
	n_instances = len(data_X)		
	n_categories = len(all_categories)


# recupera i dati dalla cartella DLI32
# legge da un dataset dove ogni categoria si trova in una cartella (1 entry per file)
def dataFromDLI32():
	
	global category_lines
	global all_categories
	global n_categories
	
	# resetta i dati
	category_lines = {}
	all_categories = []
	
	#for folder in findFiles('DLI32-2/*'):
	for folder in findFiles('TrainData/*'):
		category = folder.split('/')[-1]	# categoria = nome della cartella
		all_categories.append(category)
		category_lines[category] = []
		
		# dati da ogni file txt
		for filename in findFiles(folder+"/*.txt"):
			content = open(filename, 'r').read()
			category_lines[category].append(content)
			
	n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(lines):
	
	lens = [len(line) for line in lines]
	tensor = torch.zeros(max(lens), len(lines), n_letters)
	# The 1st axis is the sequence itself, the 2nd indexes instances in the mini-batch, the 3rd indexes elements of the character.
	
	for i, line in enumerate(lines):
		line = line.decode('utf-8')		# da stringa a utf-8
		line = unidecode_expect_nonascii(line)	# da utf-8 sostituisce caratteri "strani" in stringa ASCII
		line = unicode(line)		# ad unicode
	
		line = unicodeToAscii(line)
		
		for j, letter in enumerate(line):
			tensor[j][i][letterToIndex(letter)] = 1
	return tensor


# rimescola i dati
def shuffle():
	global index
	
	seed = 666
	numpy.random.seed(seed)
	numpy.random.shuffle(data_X)
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
	

