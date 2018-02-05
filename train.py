# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import argparse		# per i parametri da terminale
import re		# regular expression

# moduli del progetto
import data
from data import lineToTensor, findFiles
import model
from predict import predict

#n_hidden = 246
n_hidden = 64
print_every = 1

cuda = torch.cuda.is_available()	# verifica se cuda è disponibile

def categoryFromOutput(output):
	try:
		top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
		category_i = top_i[0][0]
		return data.all_categories[category_i], category_i
	except (ValueError,IndexError) as error:
		print "output:", output
		print "category_i:", category_i
		print "all_categories:", len(data.all_categories)
		raise error

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# crea due tensori input/output per la RNN
def randomTrainingPair():
    category = randomChoice(data.all_categories)
    line = randomChoice(data.category_lines[category])
    category_tensor = Variable(torch.LongTensor([data.all_categories.index(category)]))
    line_tensor = Variable(lineToTensor([line]))
    return line, category, line_tensor, category_tensor


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
	parser.add_argument('--model', default='RNN', help='modello da usare: RNN | LSTM', choices=['RNN','LSTM'])
	parser.add_argument('--dataset', metavar='dataset', help='Dataset da usare: names | dli32', choices=['names', 'dli32'])
	parser.add_argument('--epochs', type=int, default=10, metavar='epochs', help='number of epochs to train (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='lr', help='learning rate (default: 0.005)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='batch-size', help='size of mini-batch')
	parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
	parser.add_argument('--restart', action='store_true', default=False, help='restart training from an existing model')
	args = parser.parse_args()
	
	# carica dataset
	if args.dataset == 'dli32':
		data.dataFromDLI32()
	else:
		data.dataFromFiles('data/names/*.txt')
	
	start_epoch = 1
	
	# ripristina modello da file, se richiesto
	if args.restart:
		files = findFiles("results/"+args.model+"_epoch_*.pt")
		if len(files) > 0:
			files.sort()
			filename = files[-1]
			rnn = torch.load(filename)
			
			# recupera l'epoca a cui si era rimasti
			num = re.split('_|\.', filename)[-2]
			start_epoch = int(num)+1
			
			print "Modello recuperato dal file "+filename
		else:
			print "Nessun file trovato per il modello "+args.model+". Ne verrà creato uno nuovo."
			args.restart = False
	
	# instanzia rete neurale
	if not args.restart:
		if args.model == 'RNN':
			rnn = model.RNN(data.n_letters, n_hidden, data.n_categories, cuda=args.cuda)
		elif args.model == 'LSTM':
			rnn = model.LSTM(input_size=data.n_letters, hidden_size=n_hidden, output_size=data.n_categories, cuda=args.cuda)
	
	assert rnn
	
	#optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
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
		
		optimizer.zero_grad()	# azzera i gradienti
		
		data.shuffle()
		
		# mini-batch di elementi
		for step in range(1, num_batches+1):
		
			current_loss = 0
			guessed = 0
			
			#category, line, category_tensor, line_tensor = randomTrainingPair()
			line, category, line_tensor, category_tensor = data.getBatch(args.batch_size)

			# usa cuda se possibile
			if args.cuda:
				category_tensor = category_tensor.cuda()
				line_tensor = line_tensor.cuda()
		
			#line_tensor = line_tensor.view((1, len(line), -1))
			#print line_tensor
		
			output = rnn.forward(line_tensor)	# predizione tramite modello
			#print output
		
			for i in range(len(category)):
				outCategory, _ = categoryFromOutput(output[i])
				#print outCategory, category
				if outCategory == data.all_categories[category[i]]:
					guessed = guessed + 1
		
			#output = output.view((args.batch_size, data.n_categories))
			output = output.squeeze(1)
			#print output
			#print category_tensor
			
			loss = criterion(output, category_tensor)
			
			loss.backward(retain_graph=retain_graph)		# calcola gradienti (si sommano ai precedenti)
			
			# clipping del gradiente, per evitare che "esploda"
			torch.nn.utils.clip_grad_norm(rnn.parameters(), 10)
			
			optimizer.step()	# modifica pesi secondo i gradienti
			
			if step % print_every == 0:
				print('epoch: %d, step: %d/%d, loss: %f, guessed: %d / %d. (%s)' % (epoch, step, num_batches+1, loss, guessed, args.batch_size, timeSince(start)))
			
		# fine batch
		
		# stampa di fine epoca
		print('epoch: %d %d%% done. (%s)' % (epoch, float(epoch) / args.epochs * 100, timeSince(start)))
		
		# salva modello su file
		filename = '%s_epoch_%d.pt' % (args.model, epoch)
		torch.save(rnn, "results/"+filename)
		print "Model saved in file: results/%s" % (filename)
	
	# end training -------------------------------------
	
	# testing
	print predict(rnn, "Typical approaches are based on statistics of the most frequent n-grams in each language", 3, cuda=args.cuda)
	print predict(rnn, "Please see the solution details below and run the code yourself.", 3, cuda=args.cuda)
	print predict(rnn, "i file contengono una o più frasi separate da una riga vuota;", 3, cuda=args.cuda)
	print predict(rnn, "Hola, amigo, como estas?", 3, cuda=args.cuda);

