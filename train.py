# -*- coding: utf-8 -*-

# TRAIN.PY
# Trains a Recurrent Neural Network to Classify a short text in his language.
# Author: Mattia Venturini
#
# Based on Sean Robertson's work:
# https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy
import time
import argparse		# per i parametri da terminale
import re		# regular expression
import os.path

# moduli del progetto
import data
from data import lineToTensor, findFiles
import model
from predict import predict

# funzioni ---------------------------------------------------------------

def categoryFromOutput(output):
	try:
		top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
		category_i = top_i[0]
		return data.all_categories[category_i], category_i
	except (ValueError,IndexError,TypeError) as error:
		print "output:", output
		print "category_i:", category_i
		print "all_categories:", len(data.all_categories)
		raise error

def countGuessed(netOutput, categories):
	guessed = 0
	for i in range(len(categories)):
		outCategory, _ = categoryFromOutput(netOutput[i])
		if outCategory == data.all_categories[categories[i]]:
			guessed = guessed + 1
	return guessed


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

"""
# crea due tensori input/output per la RNN
def randomTrainingPair(batch_size=1):

	lines = []
	categories = []

	for i in range(batch_size):
		category = randomChoice(data.all_categories)
		category_index = data.all_categories.index(category)
		#category = random.randint(0, data.n_categories)
		line = randomChoice(data.category_lines[category])

		lines.append(line)
		categories.append(category_index)

	category_tensor = Variable(torch.LongTensor(categories))
	line_tensor = Variable(lineToTensor(lines))

	return lines, categories, line_tensor, category_tensor"""


def timeSince(since):
    now = time.time()
    s = now - since
    m = s // 60		# divisione intera (non-default in python 3)
    s -= m * 60
    return '%dm %ds' % (m, s)


# usa il Test Set per valutare il modello in base a loss totale e predizioni esatte
def test(rnn, n_test=1, loss_fn=nn.NLLLoss()):
	X = data.testX
	Y = data.testY
	guessed = 0
	loss = 0
	assert len(X) == len(Y), "Error! X and Y have different number of elements (X: %d, Y: %d)" % (len(X), len(Y))

	for i in range(n_test):
		index = random.randint(0, len(X)-1)
		lineTensor = Variable(lineToTensor([X[index]]))
		category_tensor = Variable(torch.LongTensor([Y[index]]))
		output = rnn.forward(lineTensor)
		guessed += countGuessed(output, [Y[index]])
		loss += loss_fn(output, category_tensor)
	return loss, guessed


# ------------------------- MAIN ------------------------

if __name__ == "__main__":

	# parametri da terminale
	parser = argparse.ArgumentParser(description='Language modelling at character level with a RNN (PyTorch)')
	parser.add_argument('--model', default='RNN', help='modello da usare: RNN | LSTM', choices=['RNN','LSTM','GRU'])
	parser.add_argument('--dataset', metavar='dataset', default='TrainData', help='Dataset da usare: names | dli32', choices=['names','TrainData'])
	parser.add_argument('--epochs', type=int, default=10, metavar='epochs', help='number of epochs to train (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='lr', help='learning rate (default: 0.005)')
	parser.add_argument('--batch-size', type=int, default=1, metavar='batch-size', help='size of mini-batch')
	parser.add_argument('--n-hidden', type=int, default=128, metavar='n_hidden', help='dimension of the hidden state')
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	parser.add_argument('--restart', action='store_true', default=False, help='restart training from an existing model')
	args = parser.parse_args()

	start_epoch = 1
	start_step = 1
	print_every = 100
	save_every = 5000

	# recupera dataset
	if args.dataset == 'TrainData':
		data.dataFromFiles('TrainData/*.train.utf8')
	elif args.dataset == 'names':
		data.dataFromFiles('data/names/*.txt')


	# ripristina modello da file, se richiesto
	if args.restart:
		files = findFiles("results/"+args.model+"_epoch_*.pt")
		if len(files) > 0:
			mtimes = [os.path.getmtime(f) for f in files]	# ricava data di modifica dei file
			max_idx = numpy.argmax(mtimes)	# prende quella maggiore = la più recente
			filename = files[max_idx]

			rnn = torch.load(filename)

			# recupera l'epoca a cui si era rimasti
			tokens = re.split('_|\.', filename)
			if tokens[-3] == 'checkpoint':
				start_epoch = int(tokens[-4])
				start_step = int(tokens[-2]) + 1
			else:
				start_epoch = int(tokens[-2]) + 1
				start_step = 1
			#num = re.split('_|\.', filename)[-2]
			#start_epoch = int(num)+1

			print "Modello recuperato dal file "+filename
		else:
			print "Nessun file trovato per il modello "+args.model+". Ne verrà creato uno nuovo."
			args.restart = False

	# instanzia nuova rete neurale
	if not args.restart:
		if args.model == 'RNN':
			rnn = model.RNN(data.n_letters, args.n_hidden, data.n_categories, cuda=args.cuda)
		elif args.model == 'LSTM':
			rnn = model.LSTM(input_size=data.n_letters, hidden_size=args.n_hidden, output_size=data.n_categories, cuda=args.cuda)
		elif args.model == 'GRU':
			rnn = model.GRU(input_size=data.n_letters, hidden_size=args.n_hidden, output_size=data.n_categories, cuda=args.cuda)

	assert rnn

	#optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr)
	optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
	criterion = nn.NLLLoss()

	if args.cuda:
		rnn.cuda()
		criterion.cuda()

	start = time.time()
	num_batches = data.n_instances / args.batch_size
	print "num_batches: "+str(num_batches)

	retain_graph = False
	if args.model == 'LSTM':	# non so bene il perché di tutto ciò...
		retain_graph = True

	# training ---------------------------------------
	for epoch in range(start_epoch, args.epochs + 1):

		data.shuffle()

		# mini-batch di elementi
		for step in range(start_step, (num_batches+1)):

			optimizer.zero_grad()	# azzera i gradienti
			guessed = 0

			#line, category, line_tensor, category_tensor = randomTrainingPair(args.batch_size)
			line, category, line_tensor, category_tensor = data.getBatch(args.batch_size)

			# usa cuda se possibile
			if args.cuda:
				category_tensor = category_tensor.cuda()
				line_tensor = line_tensor.cuda()

			output = rnn.forward(line_tensor)	# predizione tramite modello
			#output = output.squeeze(1)

			loss = criterion(output, category_tensor)
			loss.backward(retain_graph=retain_graph)		# calcola gradienti (si sommano ai precedenti)

			# clipping del gradiente, per evitare che "esploda"
			torch.nn.utils.clip_grad_norm(rnn.parameters(), 10.0)

			optimizer.step()	# modifica pesi secondo i gradienti

			if step % print_every == 0:
				#output = output.unsqueeze(1)
				guessed = countGuessed(output, category)	# classificazioni corrette in questo batch
				print('epoch: %d, step: %d/%d, loss: %f, guessed: %d / %d. (%s)' % (epoch, step, num_batches+1, loss, guessed, args.batch_size, timeSince(start)))

			if step % save_every == 0:
				# salva checkpoint del modello
				filename = '%s_epoch_%d_checkpoint_%d.pt' % (args.model, epoch, step)
				torch.save(rnn, "results/"+filename)
				print "Model saved in file: results/%s" % (filename)

				# testing sul checkpoint
				loss, guessed = test(rnn, n_test=100, loss_fn=criterion)
				print('testing ---> loss: %f, guessed: %d / %d' % (loss, guessed, 100))

		# fine batch

		# stampa di fine epoca
		print('epoch: %d %d%% done. (%s)' % (epoch, float(epoch) / args.epochs * 100, timeSince(start)))

		# salva modello su file
		filename = '%s_epoch_%d.pt' % (args.model, epoch)
		torch.save(rnn, "results/"+filename)
		print "Model saved in file: results/%s" % (filename)

		# testing di fine epoca ---------------------
		loss, guessed = test(rnn, n_test=100, loss_fn=criterion)
		print('testing ---> loss: %f, guessed: %d / %d' % (loss, guessed, 100))

	# end training -------------------------------------
