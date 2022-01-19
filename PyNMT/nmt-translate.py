#!/usr/bin/python3

import sys
import argparse
import random
import operator
import pickle

import torch
import torch.nn.functional as F

sys.path.insert(0,'.')
from PyNMT.Data import Data, rstrip_zeros
from PyNMT.NMT import NMTDecoder


def translate(model, data, inputfile, args):

   batch_no = 0
   with torch.no_grad():
      for src_words, sent_idx, (src_wordIDs, src_len) in data.test_batches(inputfile):
         if not args.quiet:
            print(batch_no*args.batch_size+len(src_words), end="\r", file=sys.stderr)
         batch_no += 1
   
         tgt_wordIDs, logprobs = model.translate(src_wordIDs, src_len, args.beam_size)
         # undo the sorting of sentences by length
         tgt_wordIDs = [tgt_wordIDs[i] for i in sent_idx]
         logprobs = [logprobs[i] for i in sent_idx]
         
         for swords, twordIDs, lp in zip(src_words, tgt_wordIDs, logprobs):
            if args.print_source:
               print(' '.join(swords))
            twords = data.target_words(rstrip_zeros(twordIDs))
            print(' '.join(twords))
            if args.print_probs:
               print(float((lp/(len(twords)+1)).exp()))
            if args.print_source:
               print('')
         


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Training program of the RNN-Tagger.')

   parser.add_argument('path_param', type=str,
                       help='file to which the network parameters are stored')
   parser.add_argument('path_data', type=str,
                       help='file containing the input data')
   parser.add_argument('--batch_size', type=int, default=32,
                       help='size of each batch')
   parser.add_argument('--beam_size', type=int, default=0,
                       help='size of the search beam')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU (default is GPU 0)')
   parser.add_argument("--quiet", action="store_true", default=False,
                       help="print status messages")
   parser.add_argument("--print_probs", action="store_true", default=False,
                       help="print the translation probabilities")
   parser.add_argument("--print_source", action="store_true", default=False,
                       help="print source sentences")
   args = parser.parse_args()

   if args.beam_size < 0 or args.beam_size > 1000:
      sys.exit("beam size is out of range: "+str(args.beam_size))
   if args.beam_size > 0:
      args.batch_size = 1
   elif args.batch_size < 1 or args.batch_size > 1000:
      sys.exit("batch size is out of range: "+str(args.batch_size))

   # load parameters
   data  = Data(args.path_param+".io", args.batch_size) # read the symbol mapping tables

   ### creation of the network
   with open(args.path_param+".hyper", "rb") as file:
      hyper_params = pickle.load(file)
   model = NMTDecoder(*hyper_params)
   model.load_state_dict( torch.load(args.path_param+".nmt") )   # read the model
   
   if args.gpu >= 0 and torch.cuda.is_available():
      if args.gpu >= torch.cuda.device_count():
         args.gpu = 0
      torch.cuda.set_device(args.gpu)
      model = model.cuda()

   with open(args.path_data) as inputfile:
      translate(model, data, inputfile, args)
