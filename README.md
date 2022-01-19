# rub-met21
Project for preprocessing and automatically annotating different corpora for import in webanno.

# Installation
The following dependencies should be installed when using the preprocessing and annotation scripts:

``pip install requirements.txt``

Additionally you need to install the RNNTagger from [this repo](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/).
If you do not use Windows, you can replace the rnn-tagger-german.bat with its shell counterpart (on Linux based systems).

Run src/preprocessing.py in order to get the tokenized files in the respective data folders (one token per line, sentence boundaries marked with empty line).
