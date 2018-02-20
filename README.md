# Text Language Identification #

Language identification at character level with a RNN (PyTorch).

Project by Mattia Venturini.
Master Degree in Computer Science, University of Bologna, Italy.
Natural Language Processing course.

### Summary ###

* __train.py__ - Trains a Recurrent Neural Network to Classify a short text in his language.
* __data.py__ - Loads dataset from text files.
* __model.py__ - Implements different kind of RNN.
* __predict.py__ - Tests a trained model on user input.
* __sets.py__ - Subdivide a dataset (from a text file) in train, test and validation set.
* __validation.py__ - Test a model on the validation set and print results.


### How to use ###

1. Load your dataset into the TrainData folder. You need to have a different text file for every category, with every entry at every line.
2. Rename your data files in _"<category>.<train | test | validation>.utf8"_ . Use the script _sets.py_ if you need to create three files from a single one.
3. Create a folder named _results_.
4. Launch _train.py_ with the parameters you need; you will find the trained models in the results _folder_.
5. You can test a model using a sentence of yours with the script _predict.py_ or using the entire validation set with _validation.py_.
