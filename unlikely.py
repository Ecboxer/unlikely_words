##################################################################
#Language Model
#This file identifies words that are not well-predicted by a model
##################################################################

import argparse
import csv
import os

import torch
from torch.autograd import Variable

import textData
from utils import norm_weights

parser = argparse.ArgumentParser(description='PyTorch Language Model')

#Parameters
parser.add_argument('--data', type=str, default='baum_wiz_clean',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='unlikely_words.csv',
                    help='output file for unlikely words report')
parser.add_argument('--diff', type=float, default=0.1,
                    help='threshold for determining unlikeliness')
parser.add_argument('--ignore', type=str, default='<eos>',
                    help='generated words to be ignored')
parser.add_argument('--text', type=str, default='test',
                    help='text used to assess model (train, valid, test)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature (diversity increases with arg value)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

#Set random seed for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cuda' if args.cuda else 'cpu')

if args.temperature < 1e-3:
    parser.error('--temperature has to be greater than or equal to 1e-3')

if args.text not in ['train', 'valid', 'test']:
    raise ValueError( """An invalid option for `--text` was supplied.
                     options are ['train', 'valid', 'test']""")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = textData.Corpus(args.data)
ignored = args.ignore.split()
hidden = model.init_hidden(1)
unlikely_dict = dict()

if args.text == 'train':
    corpus_eval = corpus.train
if args.text == 'valid':
    corpus_eval = corpus.valid
if args.text == 'test':
    corpus_eval = corpus.test
    
with torch.no_grad(): #Do not track history
    hits = 0
    for i, word in enumerate(corpus_eval[:-1]):
        # Compare generated word's probability to true word's probability
        input = word.view(1,1)
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(args.temperature).exp().cpu()
        word_probs = norm_weights(word_weights)
        word_idx = torch.multinomial(word_weights, 1)[0] #Index of generated word
        input.fill_(word_idx)
        true_word = corpus.dictionary.idx2word[corpus_eval[i+1]]
        gen_word = corpus.dictionary.idx2word[word_idx] #Generated word

        true_prob = word_probs[corpus_eval[i+1]]
        gen_prob = word_probs[corpus_eval[i]]
        #Report those words which the model predicts to have a probability
        #of `diff` less than the generated word
        if gen_prob - true_prob > args.diff and gen_word not in ignored:
            hits += 1
            if (true_word, gen_word) not in unlikely_dict:
                unlikely_dict[(true_word, gen_word)] = 1
            else:
                unlikely_dict[(true_word, gen_word)] += 1

print('Detected {} discrepancies with given parameters'.format(hits))

pathout = os.path.join(args.data, args.outf)
period = args.outf.find('.')
suffix = 0
while os.path.exists(pathout):
    new_outf = ''.join([args.outf[:period], str(suffix), args.outf[period:]])
    pathout = os.path.join(args.data, new_outf)
    suffix += 1
    change_fout = True

with open(pathout, 'w') as fout:
    wrt = csv.writer(fout, delimiter=',')
    wrt.writerow(['diff: {}'.format(args.diff), 'ignore: {}'.format(args.ignore), 'text: {}'.format(args.text)])
    wrt.writerow(['actual', 'generated', 'freq'])
    for key in unlikely_dict:
        wrt.writerow([key[0], key[1], unlikely_dict[key]])

print('Wrote discrepancy data to {}'.format(pathout))
