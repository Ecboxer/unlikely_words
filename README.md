# RNN for PyTorch
This repo contains scripts for training an RNN in PyTorch and identifying "unlikely words".
"Unlikely words" refers to those words a model fails to generate and which it assigns a probability of some delta > 0 less than the word it generates instead. These are identified by running a trained model over some corpus.

## Software Requirements
The code requires Python 3, PyTorch, NLTK, numpy and pandas. A GPU implementation is used by default, but a CPU implementation is included as well. Requirements are provided in `requirements.txt`.

## Example Usage
The model parameters `model.pt` were obtained by training an LSTM with two hidden layers for 40 epochs on an AWS EC2 instance. The corpus used was a concatenation of Charles Dickens' *Oliver Twist* and *A Tale of Two Cities* and Plato's *Republic* and *Symposium*. This model can be used to generate text with the generate script.  
Download the repo and install the requirements with `pip install -r requirements.txt`. Then,
```bash
# Generate 1000 words from the default model
python generate.py --data dickens_oliver_dickens_two_cities_plato_republic_plato_symposium_clean/
```
To view the generated words from the terminal
```bash
less dickens_oliver_dickens_two_cities_plato_republic_plato_symposium_clean/generated.txt
# Press q to quit the view of generated.txt
```
Data on unlikely words was written to the file `unlikely_words.csv` in the corpus folder with the following
```bash
python unlikely.py --data dickens_oliver_dickens_two_cities_plato_republic_plato_symposium_clean --diff .025
```
`unlikely.py` generates a file with the frequency of occurrences for the event of an unlikely word. I have defined such an event as having the model assign the true next word a probability of `--diff` less than that which it assigns to the generated word.  
The motivation for this metric was to compare the capacities of two or more models in a manner that would be more interpretable by an audience that may not understand a loss function or the idea of training-validation-test splits.  

## Usage
To train your own model I suggest running `preprocessing.py` which will clean a txt file and perform a training-validation-test split. `preprocessing.py` accepts the following arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           text file(s) to be processed
  --Gutenberg {y,n}     is the data in Gutenberg formatting? (default y)
  --seed SEED           random seed
  --shuffle {y,n}       shuffle the text when splitting? (default y)
  --percent_train PERCENT_TRAIN
                        share of the text to use for training (default .64)
  --percent_valid PERCENT_VALID
                        share of the text to use for validation (default .16)
  --suffix SUFFIX       suffix for clean txt file
  --lower {y,n}         change all words to lowercase (default n)
```
You will now have three files, `train.txt`, `valid.txt`, `test.txt`, in a directory with the name of your file(s) followed by `_clean` (or whichever string you supplied to `--suffix`).
To train your own model the `main.py` script accepts the following arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (GRU, LSTM, RNN_RELU, RNN_TANH)
  --emsize EMSIZE       size of word embedding (default 128)
  --nhid NHID           number of hidden units per layer (default 256)
  --nlayers NLAYERS     number of layers (default 2)
  --lr LR               initial learning rate (default 20)
  --clip CLIP           gradient clipping (default 0.25)
  --epochs EPOCHS       upper epoch limit (default 40)
  --batch_size N        batch size (default 20)
  --bptt BPTT           sequence length (default 35)
  --dropout DROPOUT     dropout applied to layers (0 := no dropout) (default
                        0.2)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save final model (default model.pt)
  --onnx-export ONNX_EXPORT
                        path to export final model in onnx format
```
*Note* If you do not supply an argument to `--save`, model parameters will overwrite the file `model.pt`.
To generate text with your model use `generate.py`.
To assess unlikely words use `unlikely.py`, which accepts the following arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --checkpoint CHECKPOINT
                        model checkpoint to use
  --outf OUTF           output file for unlikely words report
  --diff DIFF           threshold for determining unlikeliness (default 0.05)
  --ignore IGNORE       generated words to be ignored
  --text TEXT           text used to assess model (train, valid, test)
  --seed SEED           random seed
  --cuda                use CUDA
  --temperature TEMPERATURE
                        temperature (diversity increases with arg value)
  --log-interval LOG_INTERVAL
                        reporting interval
```
The probability of a word discrepancy is a function of both model accuracy and the size of your vocabulary, and so at the default value of 0.5 I have found that models generate stopwords such as "the", "and", "of", punctuation, and "EOS" (the default end of sentence token). By default `unlikely.py` ignores "EOS" since that makes up the preponderance of discrepancies.
You can find an example of unlikely words analysis in the Jupyter notebook `unlikely_analysis.ipynb`. My model has, after discounting stopwords, underrepresented "Oliver", "replied" and "gentleman".

## Acknowledgements
PyTorch's example [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) was used for the RNN architecture. Texts were downloaded from [Project Gutenberg](https://www.gutenberg.org/).  

#### TODO
Plot unlikely.py output  
Can I identify madeup words by training on several authors and running `unlikely.py` on the work of another?  
Different identification metric: Find ground truth probabilities with a word frequency distribution.
