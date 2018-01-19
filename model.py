import torch
import torch.nn as nn
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
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
	
	
	# inizializza la memoria, per cominciare una sequenza
    def initHidden(self):
    	hidden = Variable(torch.zeros(1, self.hidden_size))
    	if self.useCuda:
    		hidden = hidden.cuda()
    	return hidden
    
    
    # fa un passaggio completo di una sequenza di input
    def recurrentForward(self, line_tensor):
		hidden = self.initHidden()
		
		for i in range(line_tensor.size()[0]):
			output, hidden = self.forward(line_tensor[i], hidden)
		
		return output

