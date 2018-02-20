# -*- coding: utf-8 -*-

# VALIDATION.PY
# Test a model on the validation set and print results.
# Author: Mattia Venturini

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

import data
from data import lineToTensor
from train import categoryFromOutput

# ----------------------------- MAIN ------------------------------
if __name__ == "__main__":
    # parametri da terminale
    assert sys.argv[1], "Errore: devi indicare il modello da valutare."

    data.dataFromFiles(getData=False, getTestSet=False, getValidationSet=True)

    # matrice di confusione dove memorizzare i risultati delle classificazioni
    confusionMat = numpy.zeros((data.n_categories,data.n_categories))

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

        _, outCategory = categoryFromOutput(output[0])
        if outCategory == data.validationY[i]:
    	       guessed += 1
    	loss += float(criterion(output, category_tensor))
        confusionMat[data.validationY[i], outCategory] += 1

        if (i+1) % print_every == 0:
            print("step: %d / %d, guessed: %d / %d" % (i+1, N, guessed, i+1))

    print("loss: %f, guessed: %d / %d" % (loss / N, guessed, len(data.validationX)))

    # stampa la matrice di confusione
    print("Confusion Matrix:")
    print("\t"),
    for j in data.all_categories:
        print(j[:6]+"\t"),
    print('\n')
    for i, row in enumerate(confusionMat):
        print(data.all_categories[i][:6]+"\t"),
        for elem in row:
            print("%d\t" % int(elem)),
        print('\n')
