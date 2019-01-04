import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
import random
import os
import torch
from datetime import datetime
from gru_theano import GRUTheano

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
        
def load_Gutenberg(paths):
    """Load specified file with formatting from Gutenberg"""
    text = []
    if type(paths) == str:
        f = open(paths, 'r')
        #Read corpus into text as a list of lines
        ast_count = 0
        for line in f.readlines():
            if ast_count == 0: #Gutenberg preamble
                if line[0] == '*':
                    ast_count += 1
            elif ast_count == 1: #Book text
                if line[0] == '*': #Gutenberg post
                    ast_count += 1
                else:
                    text.append(line)
        print('Read text from %s' % paths)
        f.close()
    if type(paths) == list:
        for path in paths:
            f = open(path, 'r')
            #Read corpus into text as a list of lines
            ast_count = 0
            for line in f.readlines():
                if ast_count == 0: #Gutenberg preamble
                    if line[0] == '*':
                        ast_count += 1
                elif ast_count == 1: #Book text
                    if line[0] == '*': #Gutenberg post
                        ast_count += 1
                    else:
                        text.append(line)
            print('Read text from %s' % path)
            f.close()

    #Form corpus by joining list of lines
    corpus = ''.join(text)

    #Remove line breaks and returns
    corpus = corpus.replace('\n', ' ')
    corpus = corpus.replace('\r', '')
    corpus = corpus.replace('\\', '')

    #Remove multiple whitespace
    corpus = ' '.join(corpus.split())
    
    return corpus

def preprocessing(corpus, sentence_start_token=SENTENCE_START_TOKEN, sentence_end_token=SENTENCE_END_TOKEN):
    """Perform preprocessin on the corpus, up until word frequency
    Return tokenized sentences, sentences and an nltk FreqDist object"""
    #Split corpus into sentences
    sentences = nltk.sent_tokenize(corpus.decode('utf-8').lower())

    #Append sentence start and end tokens
    sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print('Parsed %d sentences' % len(sentences))

    #Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    #Count word frequency
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print('Found %d unique word tokens' % len(word_freq.items()))

    return tokenized_sentences, sentences, word_freq

def construct_idxs(freqdist, vocabulary_size, unknown_token=UNKNOWN_TOKEN):
    """Get the vocabulary_size^{th} most common words and build vocabulary, idx2word and word2idx"""

    vocab = freqdist.most_common(vocabulary_size-1)
    idx2word = [x[0] for x in vocab]
    idx2word.append(unknown_token)
    word2idx = dict([(w,i) for i,w in enumerate(idx2word)])
    print('Using vocabulary size %d' % vocabulary_size)
    print('The least frequent word in our vocabulary is "%s" appearing %d times' % (vocab[-1][0], vocab[-1][1]))

    return vocab, idx2word, word2idx

def replace_unknown(tokenized_sentences, sentences, word2idx, unknown_token=UNKNOWN_TOKEN):
    """Replace all words not in our vocabulary with the unknown token and return fully tokenized sentences"""
    for i,sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word2idx else unknown_token for w in sent]
        temp = np.random.randint(0, len(sentences))

    print('Example sentence: "%s"\nExample sentence after pre-processing: "%s"' % (sentences[temp], tokenized_sentences[temp]))

    return tokenized_sentences

def create_training_data(tokenized_sentences, sentences, word2idx, idx2word):
    """Return features and labels for RNN"""
    X_train = np.asarray([[word2idx[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word2idx[w] for w in sent[1:]] for sent in tokenized_sentences])

    #Training data example
    temp = np.random.randint(0, len(sentences))
    X_example, y_example = X_train[temp], y_train[temp]
    print('X:\n%s\n%s' % (' '.join([idx2word[x] for x in X_example]), X_example))
    print('y:\n%s\n%s' % (' '.join([idx2word[x] for x in y_example]), y_example))

    return X_train, y_train

def preprocessing_pipeline(path, vocab_size=0):
    """Runs the entire preprocessing pipeline
    Returns corpus, tokenized_sentences, sentences, word_freq, vocab, id2word, word2idx, X_train, y_train"""
    corpus = load_Gutenberg(path)
    tokenized_sentences, sentences, word_freq = preprocessing(corpus)
    
    if vocab_size == 0: #Make a guess of two-thirds vocabulary size
        vocab_size = int(2*word_freq.B() / 3)
    vocab, idx2word, word2idx = construct_idxs(word_freq, vocab_size)
    tokenized_sentences = replace_unknown(tokenized_sentences, sentences, word2idx)
    X_train, y_train = create_training_data(tokenized_sentences, sentences, word2idx, idx2word)

    return corpus, tokenized_sentences, sentences, word_freq, vocab, idx2word, word2idx, X_train, y_train

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)            
    return model

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print("Saved model parameters to %s." % outfile)
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))

def save_model_parameters_theano_gru(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print("Saved model parameters to %s." % outfile)

def load_model_parameters_theano_gru(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print("Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim))
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['E', 'U', 'W', 'b', 'V', 'c']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))

def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Sometimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)

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
