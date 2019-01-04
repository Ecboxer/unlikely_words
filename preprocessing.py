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
                    help='text file(s) to be processed')
parser.add_argument('--Gutenberg', choices=['y', 'n'], default='y',
                    help='is the data in Gutenberg formatting? (default y)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--shuffle', choices=['y', 'n'], default='y',
                    help='shuffle the text when splitting? (default y)')
parser.add_argument('--percent_train', type=float, default=.64,
                    help='share of the text to use for training (default .64)')
parser.add_argument('--percent_valid', type=float, default=.16,
                    help='share of the text to use for validation (default .16)')
parser.add_argument('--suffix', type=str, default='_clean',
                    help='suffix for clean txt file')
parser.add_argument('--lower', choices=['y', 'n'], default='n',
                    help='change all words to lowercase (default n)')
args = parser.parse_args()

files = args.data.split() #Get list of files
periods = [f.find('.') for f in files] #Find indices of file extensions
prefixes = [f[:periods[i]] for i,f in enumerate(files)] #Find file names
new_file = '_'.join(prefixes) + args.suffix + '.txt' #Output file name

#Write clean txt file if original is from Gutenberg
if os.path.exists(new_file):
    parser.error('output file already exists, edit --suffix or removing the file and rerun')
elif args.Gutenberg == 'y':
    write_Gutenberg(files, new_file)

#write_Gutenberg outputs one joint file
#Otherwise we have to combine them
if args.Gutenberg != 'y':
    with open(new_file, 'w') as fout:
        for f in files:
            with open(f) as fin:
                fout.write(fin.read())

#Create directory for output
os.mkdir(new_file.split('.')[0])

#Perform train-valid-test split and store files in new directory
if args.lower == 'y':
    lc = True
else:
    lc = False
Shuffle = True if args.shuffle == 'y' else False
split_file(new_file, args.percent_train, args.percent_valid, isShuffle=Shuffle, seed=args.seed, lowercase=lc)
