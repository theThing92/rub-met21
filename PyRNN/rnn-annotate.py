#!/usr/bin/python3

import sys
import argparse
import pickle
import torch
from torch.nn import functional as F

sys.path.insert(0,'.')
from PyRNN.Data import Data
from PyRNN.RNNTagger import RNNTagger
from PyRNN.CRFTagger import CRFTagger


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Annotation program of the RNN-Tagger.')

   parser.add_argument('path_param', type=str,
                       help='name of parameter file')
   parser.add_argument('path_data', type=str,
                       help='name of the file with input data')
   parser.add_argument('--crf_beam_size', type=int, default=10,
                       help='size of the CRF beam (if the system contains a CRF layer)')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU (default is GPU 0)')
   parser.add_argument("--print_probs", action="store_true", default=False,
                       help="print the tag probabilities (not possible if the tagger uses a CRF layer)")

   args = parser.parse_args()

   # load parameters
   data  = Data(args.path_param+".io")   # read the symbol mapping tables

   with open(args.path_param+".hyper", "rb") as file:
      hyper_params = pickle.load(file)
   model = CRFTagger(*hyper_params) if len(hyper_params)==10 else RNNTagger(*hyper_params)
   model.load_state_dict(torch.load(args.path_param+".rnn"))
   
   if args.gpu >= 0 and torch.cuda.is_available():
      if args.gpu >= torch.cuda.device_count():
         args.gpu = 0
      torch.cuda.set_device(args.gpu)
      model = model.cuda()

   model.eval()
   with torch.no_grad():
      for i, words in enumerate(data.sentences(args.path_data)):
         print(i, end="\r", file=sys.stderr, flush=True)
   
         # map words to numbers and create Torch variables
         fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
         fwd_charIDs = model.long_tensor(fwd_charIDs)
         bwd_charIDs = model.long_tensor(bwd_charIDs)
         
         # optional word embeddings
         word_embs = None if data.word_emb_size<=0 else model.float_tensor(data.words2vecs(words))
         
         # run the model
         if type(model) is RNNTagger:
            tagscores = model(fwd_charIDs, bwd_charIDs, word_embs)
            if args.print_probs:
               probs = F.softmax(tagscores,dim=-1)
               tagprobs, tagIDs = probs.max(dim=-1)
               tags = data.IDs2tags(tagIDs)
               for word, tag, prob in zip(words, tags, tagprobs):
                  print(word, tag, float(prob), sep="\t")
            else:
               _, tagIDs = tagscores.max(dim=-1)
               tags = data.IDs2tags(tagIDs)
               for word, tag in zip(words, tags):
                  print(word, tag, sep="\t")
         elif type(model) is CRFTagger:
            tagIDs = model(fwd_charIDs, bwd_charIDs, word_embs)
            tags = data.IDs2tags(tagIDs)
            for word, tag in zip(words, tags):
               print(word, tag, sep="\t")
         else:
            sys.exit("Error")
   
   
         print("")
