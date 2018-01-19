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
print_every = 10
plot_every = 1000
#learning_rate = 0.001

cuda = torch.cuda.is_available()	# verifica se cuda è disponibile

def categoryFromOutput(output):
	print output
	top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	category_i = top_i[0][0]
	print category_i
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

# training su un input
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

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
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	args = parser.parse_args()
	
	if args.dataset == 'dli32':
		data.dataFromDLI32()
	else:
		data.dataFromFiles()
	
	# instanzia rete neurale
	rnn = model.RNN(data.n_letters, n_hidden, data.n_categories, cuda=args.cuda)
	optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9)
	criterion = nn.NLLLoss()
	if args.cuda:
		rnn.cuda()
	
	start = time.time()
	
	# training (online) ------------------------------------
	for epoch in range(1, args.epochs + 1):
		category, line, category_tensor, line_tensor = randomTrainingPair()
		
		# usa cuda se possibile
		if args.cuda:
			category_tensor = category_tensor.cuda()
			line_tensor = line_tensor.cuda()
		
		output = rnn.recurrentForward(line_tensor)
		
		loss = criterion(output, category_tensor) 
		loss.backward()		# calcola gradienti
		
		# clipping del gradiente, per evitare che "esploda"
		torch.nn.utils.clip_grad_norm(rnn.parameters(), 0.25)
		
		optimizer.step()	# modifica pesi secondo i gradienti
		current_loss += loss

		# Print epoch number, loss, name and guess
		if epoch % print_every == 0:
			guess, guess_i = categoryFromOutput(output)
			correct = 'Yes' if guess == category else 'No (%s)' % category
			printed_input = data.unicodeToAscii(line)[0:32]+"..."
			print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, float(epoch) / args.epochs * 100, timeSince(start), loss, printed_input, guess, correct))

		# Add current loss avg to list of losses
		if epoch % plot_every == 0:
		    all_losses.append(current_loss / plot_every)
		    current_loss = 0
	# end training -------------------------------------
	
	# salva modello su file
	torch.save(rnn, 'char-rnn-classification.pt')
	print "Model saved in file: char-rnn-classification.pt"
	
	# testing
	"""for category in all_categories:
		for data in category_lines[category]:
			predict(rnn, data)"""
	
	print predict(rnn, "Typical approaches are based on statistics of the most frequent n-grams in each language", cuda=args.cuda)
	print predict(rnn, "Please see the solution details below and run the code yourself.", cuda=args.cuda)
	print predict(rnn, "i file contengono una o più frasi separate da una riga vuota;", cuda=args.cuda)
	print predict(rnn, "Hola, amigo, como estas?", cuda=args.cuda);

