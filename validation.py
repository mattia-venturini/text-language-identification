# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
from data import lineToTensor
from train import countGuessed

# ----------------------------- MAIN ------------------------------
if __name__ == "__main__":

    assert sys.argv[1], "Errore: devi indicare il modello da valutare."

    data.dataFromFiles(getData=False, getTestSet=False, getValidationSet=True)
    rnn = torch.load(sys.argv[1])
    criterion = nn.NLLLoss()

    guessed = 0
    loss = 0.0
    N = len(data.validationX)
    print_every = 100

    # validation
    for i in range(N):
    	lineTensor = Variable(lineToTensor([data.validationX[i]]), requires_grad=False)
    	category_tensor = Variable(torch.LongTensor([data.validationY[i]]), requires_grad=False)
    	output = rnn.forward(lineTensor)
    	guessed += countGuessed(output, [data.validationY[i]])
    	loss += float(criterion(output, category_tensor))

        if (i+1) % print_every == 0:
            print("step: %d / %d, guessed: %d / %d" % (i+1, N, guessed, i+1))

    print("loss: %f, guessed: %d / %d" % (loss / N, guessed, len(data.validationX)))
