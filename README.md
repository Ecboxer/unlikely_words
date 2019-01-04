# RNN for PyTorch
This repo contains scripts for training an RNN in PyTorch and identifying "unlikely words".
"Unlikely words" refers to those words a model fails to generate and which it assigns a probability of some $\delta > 0$ less than the word it generates instead. These are identified by running a trained model over some corpus.

## Software Requirements
The code requires Python 3, PyTorch, NLTK

#### TODO
Plots on unlikely.py output  
Instructions on how to run the scripts  
