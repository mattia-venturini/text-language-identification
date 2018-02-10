# -*- coding: utf-8 -*-

import argparse
import data

# ------------------------- MAIN ------------------------------
if __name__ == "__main__":

    # parametri da terminale
    parser = argparse.ArgumentParser(description='Partition a file in train, test and validation set.')
    parser.add_argument('data_file', help='file name')
    args = parser.parse_args()

    assert args.data_file, "Error: you must provide a text file to split"

    file_name = args.data_file.split('/')[-1]
    base_name = file_name.split('.')[0]     # nome senza estensione

    # file in cui scrivere
    f_train = open(base_name+".train.utf8", 'w')
    f_test = open(base_name+".test.utf8", 'w')
    f_validation = open(base_name+".validation.utf8", 'w')

    # estrae dati
    data.dataFromFiles(args.data_file, getData=True)
    data.shuffle()

    n_test = data.n_instances / 10

    # 1/10 per test set
    X, Y, _, __ = data.getBatch(n_test)
    s = '\n'.join(X)
    f_test.write(s)
    print "Test set created: " + base_name+".test.utf8"

    # 1/10 per validation set
    X, Y, _, __ = data.getBatch(n_test)
    s = '\n'.join(X)
    f_validation.write(s)
    print "Validation set created: " + base_name+".validation.utf8"

    # il resto per train set
    X = data.data_X[data.index :]
    s = '\n'.join(X)
    f_train.write(s)
    print "Train set created: " + base_name+".train.utf8"

    f_train.close()
    f_test.close()
    f_validation.close()
