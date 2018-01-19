# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import argparse		# per i parametri da terminale

# moduli del progetto
import data
from data import lineToTensor
import model
from predict import predict

n_hidden = 246
print_every = 1

cuda = torch.cuda.is_available()	# verifica se cuda è disponibile

def categoryFromOutput(output):
	top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	category_i = top_i[0][0]
	return data.all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# crea due tensori input/output per la RNN
def randomTrainingPair():
    category = randomChoice(data.all_categories)
    line = randomChoice(data.category_lines[category])
    category_tensor = Variable(torch.LongTensor([data.all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ------------------------- MAIN ------------------------

if __name__ == "__main__":
		
	# parametri da terminale
	parser = argparse.ArgumentParser(description='Language modelling at character level with a RNN (PyTorch)')
	parser.add_argument('--dataset', metavar='dataset', help='Dataset da usare: names | dli32')
	parser.add_argument('--epochs', type=int, default=1000, metavar='epochs', help='number of epochs to train (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='lr', help='learning rate (default: 0.005)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='batch-size', help='size of mini-batch')
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	args = parser.parse_args()
	
	if args.dataset == 'dli32':
		data.dataFromDLI32()
	else:
		data.dataFromFiles()
	
	# instanzia rete neurale
	rnn = model.RNN(data.n_letters, n_hidden, data.n_categories, cuda=args.cuda)
	#optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
	optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
	criterion = nn.NLLLoss()
	if args.cuda:
		rnn.cuda()
	
	start = time.time()
	
	# training ---------------------------------------
	for epoch in range(1, args.epochs + 1):
		
		optimizer.zero_grad()	# azzera i gradienti
		current_loss = 0
		guessed = 0
		
		# mini-batch di elementi
		for i in range(args.batch_size):
			category, line, category_tensor, line_tensor = randomTrainingPair()
	
			# usa cuda se possibile
			if args.cuda:
				category_tensor = category_tensor.cuda()
				line_tensor = line_tensor.cuda()
	
			output = rnn.recurrentForward(line_tensor)	# predizione tramite modello
			
			outCategory, _ = categoryFromOutput(output)
			#print outCategory, category
			if outCategory == category:
				guessed = guessed + 1
	
			loss = criterion(output, category_tensor)
			loss.backward()		# calcola gradienti (si sommano ai precedenti)
			current_loss = current_loss + loss
		# fine batch
					
		# clipping del gradiente, per evitare che "esploda"
		torch.nn.utils.clip_grad_norm(rnn.parameters(), 0.25)
		
		optimizer.step()	# modifica pesi secondo i gradienti
		
		# stampa cose
		if epoch % print_every == 0:
			print('epoch: %d %d%% (%s), loss: %.4f, guessed: %d / %d' % (epoch, float(epoch) / args.epochs * 100, timeSince(start), current_loss, guessed, args.batch_size))
	
	# end training -------------------------------------
	
	# salva modello su file
	torch.save(rnn, 'char-rnn-classification.pt')
	print "Model saved in file: char-rnn-classification.pt"
	
	# testing
	print predict(rnn, "Typical approaches are based on statistics of the most frequent n-grams in each language", cuda=args.cuda)
	print predict(rnn, "Please see the solution details below and run the code yourself.", cuda=args.cuda)
	print predict(rnn, "i file contengono una o più frasi separate da una riga vuota;", cuda=args.cuda)
	print predict(rnn, "Hola, amigo, como estas?", cuda=args.cuda);

