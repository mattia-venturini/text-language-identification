# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import argparse

import model
import data
from data import lineToTensor

# Just return an output given a line
def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

# deduce le prime n categorie più probabili
def predict(rnn, line, n_predictions=1):
	#print lineToTensor(line)
	output = evaluate(rnn, Variable(lineToTensor(line)))
	#print output
	
	# Get top N categories
	topv, topi = output.data.topk(n_predictions, 1, True)
	predictions = []
	
	for i in range(n_predictions):
		value = topv[0][i]
		category_index = topi[0][i]
		print('(%.2f) %s' % (value, data.all_categories[category_index]))
		predictions.append([value, data.all_categories[category_index]])

	return predictions


# ----------------------- MAIN ------------------------------
if __name__ == '__main__':
	
	# parametri da linea di comando
	parser = argparse.ArgumentParser(description='Language modelling at character level with a RNN (PyTorch)')
	parser.add_argument('target', help='String to predict')
	parser.add_argument('--dataset', metavar='dataset', help='Dataset da usare: names | dli32')
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	args = parser.parse_args()
	
	if args.dataset == 'dli32':
		data.dataFromDLI32()
	else:
		data.dataFromFiles()
	
	rnn = torch.load('char-rnn-classification.pt')
	predict(rnn, sys.argv[1], n_predictions=2)
