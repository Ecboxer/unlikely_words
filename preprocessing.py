#######################################################################
#Preprocessing
#This file creates a directory with training, validation and test files
#######################################################################

import argparse
import os
import random

from utils import write_Gutenberg, split_file

parser = argparse.ArgumentParser(description='Preprocessing for PyTorch Language Model')

#Parameters
parser.add_argument('--data', type=str, default='baum_wiz.txt',
                    help='text file to be processed')
parser.add_argument('--Gutenberg', choices=['y', 'n'], default='y',
                    help='is the data in Gutenberg formatting?')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--shuffle', choices=['y', 'n'], default='y',
                    help='shuffle the text when splitting?')
parser.add_argument('--percent_train', type=float, default=.64,
                    help='share of the text to use for training')
parser.add_argument('--percent_valid', type=float, default=.16,
                    help='share of the text to use for validation')
parser.add_argument('--suffix', type=str, default='_clean',
                    help='suffix for clean txt file')
parser.add_argument('--lower', choices=['y', 'n'], default='n',
                    help='change all words to lowercase')
args = parser.parse_args()

period = args.data.find('.')
new_file = ''.join([args.data[:period], args.suffix, args.data[period:]])

#Write clean txt file if original is from Gutenberg
if os.path.exists(new_file):
    parser.error('output file already exists, try changing --suffix or removing the file and rerunning')
elif args.Gutenberg == 'y':
    write_Gutenberg(args.data, suffix=args.suffix)

#Perform train-valid-test split and store files in new directory
if args.Gutenberg == 'y':
    file_name = new_file
else:
    file_name = args.data

#Create directory for output
os.mkdir(file_name.split('.')[0])

if args.lower == 'y':
    lc = True
else:
    lc = False
Shuffle = True if args.shuffle == 'y' else False
split_file(file_name, args.percent_train, args.percent_valid, isShuffle=Shuffle, seed=args.seed, lowercase=lc)
