import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# RNN (Recurrent Neural Network)
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, cuda=False):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		"""self.layer1 = nn.Sequential(
			nn.Linear(input_size + hidden_size, hidden_size),
			nn.LeakyReLU(0.2))"""
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=-1)

		self.useCuda = cuda


	# fa un passaggio ricorsivo
    def step(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        #combined = self.layer1(combined)	# NEW
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden


    # fa un passaggio completo di una sequenza di input
    def forward(self, line_tensor):
		batch_size = line_tensor.size()[1]
		hidden = self.initHidden(batch_size)

		for i in range(line_tensor.size()[0]):
			output, hidden = self.step(line_tensor[i], hidden)

		return output.view(batch_size, -1)


	# inizializza la memoria, per cominciare una sequenza
    def initHidden(self, size=1):
    	hidden = Variable(torch.zeros(size, self.hidden_size))
    	if self.useCuda:
    		hidden = hidden.cuda()
    	return hidden


# LSTM (Long-Short Term Memory)
# utilizza una LSTM di pytorch e la estende con qualche funzione utile
# (tratto da http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
class LSTM(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, cuda=False):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.useCuda = cuda

		self.lstm = nn.LSTM(input_size, hidden_size)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_size, output_size)


	def init_hidden(self, size=1):
		# Before we've done anything, we don't have any hidden state.
		# Refer to the Pytorch documentation to see exactly why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		hidden = torch.zeros(1, size, self.hidden_size)
		hidden2 = torch.zeros(1, size, self.hidden_size)
		if self.useCuda:
			hidden = hidden.cuda()
			hidden2 = hidden2.cuda()
		return (Variable(hidden), Variable(hidden2))	# boh, ne vuole 2


	def forward(self, line_tensor):
		self.hidden = self.init_hidden(size=line_tensor.size()[1])

		if False:
			# tutta la sequenza in una volta sola
			lstm_out, self.hidden = self.lstm(line_tensor, self.hidden)
			lstm_out = lstm_out[-1]		# prende solo la predizione alla fine della sequenza
		else:
			# itera sulla sequenza
			line_tensor = line_tensor.unsqueeze(1)
			for i in range(line_tensor.size()[0]):
				lstm_out, self.hidden = self.lstm(line_tensor[i], self.hidden)
			lstm_out = lstm_out.squeeze(0)

		tag_space = self.hidden2tag(lstm_out)
		tag_space = F.log_softmax(tag_space, dim=-1)
		return tag_space


# GRU (Gated Recurrent Unit)
class GRU(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, cuda=False):
		super(GRU, self).__init__()
		self.hidden_size = hidden_size
		self.useCuda = cuda

		self.gru = nn.GRU(input_size, hidden_size)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_size, output_size)


	def init_hidden(self, size=1):
		# Before we've done anything, we don't have any hidden state.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		hidden = torch.zeros(1, size, self.hidden_size)
		if self.useCuda:
			hidden = hidden.cuda()
		return Variable(hidden)


	def forward(self, line_tensor):
		self.hidden = self.init_hidden(size=line_tensor.size()[1])

		# itera sulla sequenza
		line_tensor = line_tensor.unsqueeze(1)
		for i in range(line_tensor.size()[0]):
			out, self.hidden = self.gru(line_tensor[i], self.hidden)
		out = out.squeeze(0)

		tag_space = self.hidden2tag(out)
		tag_scores = F.log_softmax(tag_space, dim=-1)
		return tag_scores
