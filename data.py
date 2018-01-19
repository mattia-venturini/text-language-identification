# -*- coding: utf-8 -*-

import torch
import glob
import unicodedata
import string
from io import open
from unidecode import unidecode_expect_nonascii		# per sostituire caratteri non-ASCII

# variabili globali
category_lines = {}
all_categories = []
n_categories = 0

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
def dataFromFiles(target='TrainData/*.utf8'):
	
	global category_lines
	global all_categories
	global n_categories
	
	# Build the category_lines dictionary, a list of lines per category
	category_lines = {}
	all_categories = []
	#for filename in findFiles('data/names/*.txt'):
	for filename in findFiles(target):
		category = filename.split('/')[-1].split('.')[0]
		all_categories.append(category)
		lines = readLines(filename)
		category_lines[category] = lines

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
def lineToTensor(line):
	line = line.decode('utf-8')		# da stringa a utf-8
	line = unidecode_expect_nonascii(line)	# da utf-8 sostituisce caratteri "strani" in stringa ASCII
	line = unicode(line)		# ad unicode
	
	line = unicodeToAscii(line)
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor

