import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import textData
import model

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='baum_wiz_clean',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (GRU, LSTM, RNN_RELU, RNN_TANH)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embedding')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 := no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export final model in onnx format')
args = parser.parse_args()

#Set random seed for reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cuda' if args.cuda else 'cpu')

##################
#Load data
##################

corpus = textData.Corpus(args.data)

#Starting from sequential data, arrange the dataset into columns.
#Columns are treated as independent by the model, allowing for
#more efficient batch processing.

def batchify(data, bsz):
    #Determine how cleanly the dataset can be divided into bsz parts
    nbatch = data.size(0) // bsz
    #Trim extra elements that do not fit cleanly
    data = data.narrow(0,0,nbatch*bsz)
    #Evenly divide the data across bsz batches
    data = data.view(bsz,-1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

##################
#Build model
##################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

#################
#Training code
#################

def repackage_hidden(h):
    """Wrap hidden states in new tensors, to detach from their history"""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

#get_batch subdivides source data into chunks of length args.bptt.
#Subdivision id along dimension 0, corresponding to the seq_len
#dimension of the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    #Activate evaluation, which disables dropout
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def train():
    #Activate training, which enables dropout
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        #Detach the hidden state from how it was produced previously
        #Otherwise, the model would try to backprop to the start of the dataset
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        #`clip_grad_norm` addresses the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | pp1 {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, lr,
                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

#Loop over epochs
lr = args.lr
best_val_loss = None

#To break out of training early, hit Ctrl + C
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        #Save the model if validation loss has improved
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            #Perform simulated annealing if no improvement is seen on the validation data
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

#Load best saved model
with open(args.save, 'rb') as f:
    model = torch.load(f)
    #RNN parameters are not a continuous chunk of memory
    #Turning them into such speeds up the forward pass
    model.rnn.flatten_parameters()

#Run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    #Export model in ONNX format
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
