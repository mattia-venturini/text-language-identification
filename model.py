import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# RNN (Recurrent Neural Network)
class RNN(nn.Module):
	
    def __init__(self, input_size, hidden_size, output_size, cuda=False):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
        self.useCuda = cuda
	
	
	# fa un passaggio ricorsivo
    def step(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
        
        
    # fa un passaggio completo di una sequenza di input
    def forward(self, line_tensor):
		hidden = self.initHidden()
		
		for i in range(line_tensor.size()[0]):
			output, hidden = self.step(line_tensor[i], hidden)
		
		return output
	
	
	# inizializza la memoria, per cominciare una sequenza
    def initHidden(self):
    	hidden = Variable(torch.zeros(1, self.hidden_size))
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
		
		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_size.
		self.lstm = nn.LSTM(input_size, hidden_size)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_size, output_size)
		self.hidden = self.init_hidden()
		

	def init_hidden(self):
		# Before we've done anything, we don't have any hidden state.
		# Refer to the Pytorch documentation to see exactly why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		hidden = torch.zeros(1, 1, self.hidden_size)
		if self.useCuda:
			hidden = hidden.cuda()
		return (Variable(hidden), Variable(hidden))	# boh, ne vuole 2

	def forward(self, line_tensor):
		
		#embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(line_tensor.view(len(line_tensor), 1, -1), self.hidden)
		tag_space = self.hidden2tag(lstm_out.view(len(line_tensor), -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		
		out = Variable(torch.zeros((1, tag_scores.size(1))))	# output dell'ultimo passaggio
		out[0] = tag_scores[-1]
		return out


