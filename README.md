# RNN for PyTorch
This repo contains scripts for training an RNN in PyTorch and identifying "unlikely words".
"Unlikely words" refers to those words a model fails to generate and which it assigns a probability of some $\delta > 0$ less than the word it generates instead. These are identified by running a trained model over some corpus.

## Software Requirements
The code requires Python 3, PyTorch, NLTK, numpy and pandas. A GPU implementation is used by default, but a CPU implementation is included as well. Requirements are provided in `requirements.txt`.

## Example Usage
The model parameters `model.pt` were obtained by training an LSTM with two hidden layers for 40 epochs on an AWS EC2 instance. The corpus used was a concatenation of Charles Dickens' *Oliver Twist* and *A Tale of Two Cities* and Plato's *Republic* and *Symposium*. This model can be used to generate text with the generate script.  
Download the repo and install the requirements with `pip install -r requirements.txt`. Then,
```python
# Generate 1000 words from the default model
python generate.py --data dickens_oliver_dickens_two_cities_plato_republic_plato_symposium_clean/
```
To view the output from the terminal
```python
less dickens_oliver_dickens_two_cities_plato_republic_plato_symposium_clean/generated.txt
# Press q to stop viewing generated.txt
```
To view 
## Acknowledgements
PyTorch's example [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) was used for the RNN architecture. Texts were downloaded from [Project Gutenberg](https://www.gutenberg.org/).  

#### TODO
Plots on unlikely.py output  
Instructions on how to run the scripts  
