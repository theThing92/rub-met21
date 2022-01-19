#!/usr/bin/python3

import sys
import argparse
import random
import operator
import numpy
import pickle

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

sys.path.insert(0,'.')
from PyNMT.Data import Data, rstrip_zeros
from PyNMT.NMT import NMTDecoder


def build_optimizer(optim, model, learning_rate):
   optimizer = {
      'sgd':      torch.optim.SGD,
      'rmsprop':  torch.optim.RMSprop,
      'adagrad':  torch.optim.Adagrad,
      'adadelta': torch.optim.Adadelta,
      'adam':     torch.optim.Adam
   }
   return optimizer[optim](model.parameters(), lr=learning_rate)


def process_batch(data, batch, model, optimizer=None):

   (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch

   training_mode = optimizer is not None
   model.train(training_mode)
      
   # add boundary symbols to target sequence
   tgt_wordIDs = model.long_tensor(tgt_wordIDs)
   boundaries = model.zero_long_tensor(tgt_wordIDs.size(0),1)
   tgt_wordIDs = torch.cat((boundaries, tgt_wordIDs, boundaries), dim=-1)

   scores = model(src_wordIDs, src_len, tgt_wordIDs[:,:-1])

   # flatten the first dimension of the tensors and compute the loss
   scores = scores.view(-1,scores.size(2))
   wordIDs = tgt_wordIDs[:,1:].contiguous().view(-1)
   loss = F.cross_entropy(scores, wordIDs)

   # compute gradient and perform weight updates
   if training_mode:
      optimizer.zero_grad()
      loss.backward()
      if args.grad_threshold > 0.0:
         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_threshold)
      optimizer.step()

   return float(loss)


def training(args):

   random.seed( args.random_seed )

   data = Data(args.path_train_src, args.path_train_tgt,
               args.path_dev_src, args.path_dev_tgt,
               args.max_src_vocab_size, args.max_tgt_vocab_size,
               args.max_len, args.batch_size)
   data.save_parameters( args.path_param+".io" )

   hyper_params = data.src_vocab_size, data.tgt_vocab_size, args.word_emb_size, \
      args.enc_rnn_size, args.dec_rnn_size, args.enc_depth, args.dec_depth, \
      args.dropout_rate, args.emb_dropout_rate, args.tie_embeddings
   
   with open(args.path_param+".hyper", "wb") as file:
      pickle.dump(hyper_params, file)

   ### creation of the network
   model = NMTDecoder(*hyper_params)
   
   if args.gpu >= 0:
      model = model.cuda()

   optimizer = build_optimizer( args.optimizer, model, args.learning_rate )
   scheduler = StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)

   ### training loop ##################################################
   
   loss_sum = 0
   best_result = None
   for batch_no, batch in enumerate(data.train_batches(), 1):
      print(batch_no,end="\r", file=sys.stderr, flush=True)

      loss = process_batch(data, batch, model, optimizer)
      loss_sum += loss

      ### Is it time to evaluate on development data? ###
      if batch_no % args.eval_interval == 0:

         ### translate a few training sentences #######################
         print("translation examples", file=sys.stderr)
         (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch

         tgtIDs, tgt_logprobs = model.translate(src_wordIDs, src_len)

         # print the source sentence, translation and reference translation
         for i in range(min(3,len(src_wordIDs))):
            words = data.source_words(rstrip_zeros(src_wordIDs[i]))
            print("src:", ' '.join(words), file=sys.stderr)
            words = data.target_words(rstrip_zeros(tgt_wordIDs[i]))
            print("ref:", ' '.join(words), file=sys.stderr)
            words = data.target_words(rstrip_zeros(tgtIDs[i].tolist()))
            print("tgt:", ' '.join(words), file=sys.stderr)
            print('', file=sys.stderr)

         print("Training Loss:", loss_sum/args.eval_interval, file=sys.stderr, flush=True)
         loss_sum = 0.0
         scheduler.step()

         ### evaluation on development data ###########################
         if args.eval_acc:
            # evaluation based on exact match accuracy
            print("Evaluation on dev data", file=sys.stderr)
            dev_batch_no = 0; correct = 0; all = 0
            for batch in data.dev_batches():
               dev_batch_no += 1
               print(dev_batch_no, end="\r", file=sys.stderr)
   
               (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch
               tgtIDs, tgt_logprobs = model.translate(src_wordIDs, src_len)
               for i in range(len(tgt_wordIDs)):
                  all += 1
                  pred_words = ' '.join(data.target_words(tgtIDs[i]))
                  correct_words =  ' '.join(data.target_words(tgt_wordIDs[i]))
                  if pred_words == correct_words:
                     correct += 1
            acc = correct*100/all
            print("Accuracy: %.2f"%(acc), flush=True)
   
            if best_result is None or best_result < acc:
               best_result = acc
               print("storing parameters", file=sys.stderr)
               if model.on_gpu():
                  model = model.cpu()
                  torch.save( model.state_dict(), args.path_param+".nmt" )
                  model = model.cuda()
               else:
                  torch.save( model.state_dict(), args.path_param+".nmt" )
         else:
            # evaluation based on loss
            print("Evaluation on dev data", file=sys.stderr)
            dev_batch_no = 0
            for batch in data.dev_batches():
               dev_batch_no += 1
               print(dev_batch_no, end="\r", file=sys.stderr)
               loss = process_batch(data, batch, model)
               loss_sum += loss
            print("Dev Loss:", loss_sum/dev_batch_no, flush=True)

            if best_result is None or best_result > loss_sum:
               best_result = loss_sum
               print("storing parameters", file=sys.stderr)
               if model.on_gpu():
                  model = model.cpu()
                  torch.save( model.state_dict(), args.path_param+".nmt" )
                  model = model.cuda()
               else:
                  torch.save( model.state_dict(), args.path_param+".nmt" )
            loss_sum = 0.0

      # terminate after a fixed number of batches
      if batch_no >= args.num_batches:
         break
      


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Training program of the RNN-Tagger.')

   parser.add_argument('path_train_src', type=str,
                       help='file containing the source training data')
   parser.add_argument('path_train_tgt', type=str,
                       help='file containing the target training data')
   parser.add_argument('path_dev_src', type=str,
                       help='file containing the source development data')
   parser.add_argument('path_dev_tgt', type=str,
                       help='file containing the target development data')
   parser.add_argument('path_param', type=str,
                       help='file to which the network parameters are stored')
   parser.add_argument('--word_emb_size', type=int, default=100,
                       help='size of the word embedding vectors')
   parser.add_argument('--enc_rnn_size', type=int, default=400,
                       help='size of the hidden state of the RNN encoder')
   parser.add_argument('--dec_rnn_size', type=int, default=400,
                       help='size of the hidden state of the RNN decoder')
   parser.add_argument('--enc_depth', type=int, default=1,
                       help='number of encoder BiLSTM layers')
   parser.add_argument('--dec_depth', type=int, default=1,
                       help='number of decoder LSTM layers')
   parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='dropout rate')
   parser.add_argument('--emb_dropout_rate', type=float, default=0.0,
                       help='dropout rate for embeddings')
   parser.add_argument('--optimizer', type=str, default='sgd', 
                       choices=['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam'],
                       help='seletion of the optimizer')
   parser.add_argument('--learning_rate', type=float, default=1.0,
                       help='the learning rate')
   parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                       help='the learning rate multiplier applied after each evaluation')
   parser.add_argument('--grad_threshold', type=float, default=1.0,
                       help='gradient clipping threshold')
   parser.add_argument('--max_src_vocab_size', type=int, default=0,
                       help='maximal number of words in the source vocabulary')
   parser.add_argument('--max_tgt_vocab_size', type=int, default=0,
                       help='maximal number of words in the target vocabulary')
   parser.add_argument('--batch_size', type=int, default=32,
                       help='size of each batch')
   parser.add_argument('--num_batches', type=int, default=100000,
                       help='total number of batches to train on')
   parser.add_argument('--eval_interval', type=int, default=5000,
                       help='number of batches before model is evaluated and saved')
   parser.add_argument('--random_seed', type=int, default=32,
                       help='seed for the random number generators')
   parser.add_argument('--max_len', type=int, default=50,
                       help='maximal sentence length')
   parser.add_argument('--gpu', type=int, default=-1,
                       help='selection of the GPU (default is CPU)')
   parser.add_argument("--eval_acc", action="store_true", default=False,
                       help="evaluate based on exact match accuracy rather than loss")
   parser.add_argument("--tie_embeddings", action="store_true", default=False,
                       help="Decoder input and output embeddings are tied")
   args = parser.parse_args()

   if args.gpu >= 0:
      if not torch.cuda.is_available():
         sys.exit("Sorry, no GPU available. Please drop the gpu option.")
      elif args.gpu >= torch.cuda.device_count():
         sys.exit("Sorry, given GPU index was too large. Choose a different GPU.")
      else:
         torch.cuda.set_device(args.gpu)
         torch.cuda.manual_seed( args.random_seed )
   else:
      torch.manual_seed( args.random_seed )

   training(args)
