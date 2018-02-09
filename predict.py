# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import argparse

import model
import data
from data import lineToTensor


# deduce le prime n categorie più probabili
def predict(rnn, line, n_predictions=1, cuda=False):
	#print lineToTensor(line)
	lineTensor = Variable(lineToTensor([line]))
	if cuda:
		lineTensor = lineTensor.cuda()

	output = rnn.forward(lineTensor)
	#print output

	# Get top N categories
	topv, topi = output.data.topk(k=n_predictions, dim=-1, largest=True)
	predictions = []

	for i in range(n_predictions):
		value = topv[0][i]
		category_index = topi[0][i]
		#print('(%.2f) %s' % (value, data.all_categories[category_index]))
		predictions.append([value, data.all_categories[category_index]])

	return predictions


# ----------------------- MAIN ------------------------------
if __name__ == '__main__':

	# parametri da linea di comando
	parser = argparse.ArgumentParser(description='Language modelling at character level with a RNN (PyTorch)')
	parser.add_argument('text', help='String to predict')
	parser.add_argument('--model', default='char-nn-classification.pt', help='file to load for the model')
	parser.add_argument('--dataset', metavar='dataset', help='Dataset da usare: names | dli32')
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	args = parser.parse_args()

	if args.dataset == 'dli32':
		data.dataFromDLI32()
	elif args.dataset == 'TrainData':
		data.dataFromFiles('TrainData/*.utf8', getData=False)
	elif args.dataset == 'names':
		data.dataFromFiles('data/names/*.txt', getData=False)

	rnn = torch.load(args.model)	# modello da file
	if args.cuda:
		rnn.cuda()

	predictions = predict(rnn, args.text, n_predictions=2, cuda=args.cuda)	# valuta da modello

	for p in predictions:	# stampa
		print('(%.2f) %s' % (p[0], p[1]))
