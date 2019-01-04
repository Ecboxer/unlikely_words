import csv
import numpy as np
import nltk
import time
import sys
import operator
import io
import random
import os
import torch
from datetime import datetime

SENTENCE_START_TOKEN = "SOS"
SENTENCE_END_TOKEN = "EOS"
UNKNOWN_TOKEN = "UT"

def write_Gutenberg(files, new_file):
    """Write clean txt file from file(s) with Gutenberg formatting"""
    for f in files:
        with open(f, 'r', encoding='utf-8') as fin, \
              open(new_file, 'a') as fout:
            state = 0
            n_div = sum(1 for line in fin if line[:3] == '***')
            if n_div > 0: #Condition for Gutenberg formatting
                fin.seek(0)
                ast_count = 0
                for line in fin:
                    if ast_count == 0: #Gutenberg preamble
                        if line[:3] == '***':
                            ast_count += 1
                    elif ast_count == 1: #Text
                        if line[:3] == '***': #Gutenberg post
                            ast_count += 1
                        else:
                            fout.write(line)
                print('Read {}\nWrote text to {}'.format(f, new_file))
            else:
                print('{} did not conform to expected formatting'.format(f))        

def unlikely_words(model, index_to_word, word_to_index, X, delta_min=.5, sentence_start_token=SENTENCE_START_TOKEN, verbose=False):
    """Return those words from y that were least likely to appear according to the model"""
    X_pred_probs = [model.predict(sent) for sent in X]
    X_pred = [[word_to_index[sentence_start_token]] + [j.argmax() for j in X_pred_probs[i]][:-1] for i in range(len(X_pred_probs))]
    prob_diff = [np.asarray([X_pred_probs[i][j].max() - X_pred_probs[i][j][X[i][j]] for j in range(len(X_pred_probs[i])-1)]) for i in range(len(X_pred_probs))]
    #Only those probabilities meeting the condition sent_diff > delta_min
    prob_diff_exc = [(sent_diff > delta_min) * sent_diff for sent_diff in prob_diff]
    #Sum of prob_diff_exc to check that at least one word meets the condition
    prob_diff_sum = [prob_diff_exc[i].sum() for i in range(len(prob_diff_exc))]
    #Indices of the greatest deviation between actual and predicted, greater than delta_min
    ind_exc = [prob_diff_exc[i].argmax() if prob_diff_sum[i] != 0. else 0. for i in range(len(prob_diff_sum))]
    
    if verbose == True:
        for i in range(len(X_pred)):
            if not ind_exc[i] == 0:
                print_sentence(X[i], index_to_word)
                print('Predicted word: %s' % index_to_word[X_pred[i][ind_exc[i]]])
                print('Actual word: %s' % index_to_word[X[i][ind_exc[i]]])
    return ind_exc

def split_file(file, percent_train=0.64, percent_valid=0.16, isShuffle=True, seed=1111, sentence_end_token=SENTENCE_END_TOKEN, lowercase=False):
    """Splits a file into 3 from given `percentage_` values"""
    random.seed(seed)
    folder = file.split('.')[0]
    with open(file, 'r', encoding='utf-8') as fin, \
         open(os.path.join(folder, 'train.txt'), 'w') as foutTrain, \
         open(os.path.join(folder, 'valid.txt'), 'w') as foutValid, \
         open(os.path.join(folder, 'test.txt'), 'w') as foutTest:

        #Change division from lines to sentences
        corpus = fin.read()
        corpus = corpus.replace('\n', ' ')
        corpus = corpus.replace('\r', '')
        corpus = corpus.replace('\\', '')

        if lowercase:
            sentences = nltk.sent_tokenize(corpus.lower())
        else:
            sentences = nltk.sent_tokenize(corpus)
        sentences = ['{} {} '.format(x, sentence_end_token) for x in sentences]
        n_lines = len(sentences)
        print('Parsed {} sentences'.format(n_lines))

        n_train = int(n_lines * percent_train)
        n_valid = int(n_lines * percent_valid)
        n_test = n_lines - n_train - n_valid

        i = 0
        j = 0
        k = 0
        for sentence in sentences:
            while True:
                r = random.random() if isShuffle else 0
                if (i < n_train and r < percent_train):
                    foutTrain.write(sentence)
                    i += 1
                    break
                elif (j < n_valid and r < percent_train + percent_valid):
                    foutValid.write(sentence)
                    j += 1
                    break
                else:
                    foutTest.write(sentence)
                    k += 1
                    break
    print('Wrote training data to {}\nWrote validation data to {}\nWrote test data to {}'.format('/'.join([folder, 'train.txt']), '/'.join([folder, 'valid.txt']), '/'.join([folder, 'test.txt'])))

def norm_weights(weights):
    """Take a tensor of weights and normalize so that it sums to 1"""
    return weights / weights.sum()
