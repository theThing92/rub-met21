
import sys
from collections import Counter
import pickle
import numpy

pad_string = '<pad>'
unk_string = '<unk>'

         
### helper functions ###############################################
         
def build_dict(file_name, max_vocab_size):
   """ 
   reads a list of sentences from a file and returns
   - a dictionary which maps the most frequent words to indices and
   - a table which maps indices to the most frequent words
   """
   
   word_freq = Counter()
   with open(file_name) as file:
      for line in file:
         word_freq.update(line.split())
         
   if max_vocab_size <= 0:
      max_vocab_size = len(word_freq)
            
   words, _ = zip(*word_freq.most_common(max_vocab_size))
   # ID of pad_string must be 0
   words = [pad_string, unk_string] + list(words)
   word2ID = {w:i for i,w in enumerate(words)}
   
   return word2ID, words


def pad_batch(batch):
   """ pad sequences in batch with 0s to obtain sequences of identical length """
   
   seq_len = list(map(len, batch))
   max_len = max(seq_len)
   padded_batch = [seq + [0]*(max_len-len(seq)) for seq in batch]
   return padded_batch, seq_len


def build_train_batches(big_batch, batch_size):
   # sort sentence pairs by source sentence length
   big_batch.sort(key=lambda x: len(x[0]), reverse=True)

   # extract the mini-batches
   for i in range(0, len(big_batch), batch_size):
      src_vecs, tgt_vecs = zip(*big_batch[i:i+batch_size])
      yield pad_batch(src_vecs), pad_batch(tgt_vecs)

      
def rstrip_zeros(wordIDs):
   """ removes trailing padding symbols """
   wordIDs = list(wordIDs)
   if 0 in wordIDs:
      wordIDs = wordIDs[:wordIDs.index(0)]
   return wordIDs

     
def words2IDs(words, word2ID):
   """ maps a list of words to a list of IDs """
   
   unkID = word2ID[unk_string]
   return [word2ID.get(w, unkID) for w in words]

   
### class Data ####################################################

class Data(object):
   """ class for data preprocessing """

   def __init__(self, *args):
      if len(args) == 2:
         # Initialisation for translation
         self.init_test(*args)
      else:
         # Initialisation for training
         self.init_train(*args)
   
         
   ### functions needed during training ##########################

   def init_train(self, path_train_src, path_train_tgt, path_dev_src, path_dev_tgt,
                  max_src_vocab_size, max_tgt_vocab_size, max_len, batch_size):
      """ reads the training and development data and generates the mapping tables """
      
      self.max_len    = max_len
      self.batch_size = batch_size
      
      self.path_train_src = path_train_src
      self.path_train_tgt = path_train_tgt
      self.path_dev_src   = path_dev_src
      self.path_dev_tgt   = path_dev_tgt

      self.src2ID, self.ID2src = build_dict(self.path_train_src, max_src_vocab_size)
      self.tgt2ID, self.ID2tgt = build_dict(self.path_train_tgt, max_tgt_vocab_size)
      self.src_vocab_size = len(self.ID2src)
      self.tgt_vocab_size = len(self.ID2tgt)


   def train_batches(self):
      return self.batches()
      
   def dev_batches(self):
      return self.batches(train=False)

   def batches(self, train=True):
      """ yields a sequence of (train or dev) batches """

      if train:
         src_file = open(self.path_train_src if train else self.path_dev_src)
         tgt_file = open(self.path_train_tgt if train else self.path_dev_tgt)
      else:
         src_file = open(self.path_dev_src)
         tgt_file = open(self.path_dev_tgt)

      # We read several batches at once and sort them by length
      # to improve efficiency by creating batches with less diverse length
      num_batches_in_big_batch = 5
      big_batch_size = self.batch_size * num_batches_in_big_batch
      big_batch = []
      while True:
         for src_line, tgt_line in zip(src_file, tgt_file):
            srcIDs = words2IDs(src_line.split(), self.src2ID)
            tgtIDs = words2IDs(tgt_line.split(), self.tgt2ID)

            # filter out very long sentences
            if self.max_len and max(len(srcIDs), len(tgtIDs)) > self.max_len:
               continue
            
            big_batch.append((srcIDs, tgtIDs))

            if len(big_batch) == big_batch_size:
               yield from build_train_batches(big_batch, self.batch_size)
               big_batch = []

         if train:
            # reread the two files
            src_file.seek(0)
            tgt_file.seek(0)
         else:
            # yield last batches
            yield from build_train_batches(big_batch, self.batch_size)
            break  # terminate
            

   def save_parameters(self, filename):
      """ save the module's parameters to a file """
      all_params = (self.ID2src, self.ID2tgt)
      with open(filename, "wb") as file:
         pickle.dump(all_params, file)

      
   ### functions needed for translation ############################

   def init_test(self, filename, batch_size):
      """ load parameters from a file """

      self.batch_size = batch_size
      with open(filename, "rb") as file:
         self.ID2src, self.ID2tgt = pickle.load(file)
         self.src2ID = {w:i for i,w in enumerate(self.ID2src)}
         self.tgt2ID = {w:i for i,w in enumerate(self.ID2tgt)}

   def build_test_batch(self, batch):
      batch_IDs = [words2IDs(srcWords, self.src2ID) for srcWords in batch]
      result = sorted(enumerate(batch_IDs), key=lambda x: len(x[1]), reverse=True)
      orig_sent_pos, sorted_batch_IDs = zip(*result)

      new_sent_pos = list(zip(*sorted(enumerate(orig_sent_pos), key=lambda x: x[1])))[0]

      return batch, new_sent_pos, pad_batch(sorted_batch_IDs)

   def test_batches(self, file):
      """ yields the next batch of test sentences """

      batch = []
      for line in file:
         srcWords = line.split()
         batch.append(srcWords)
         if len(batch) == self.batch_size:
            yield self.build_test_batch(batch)
            batch = []

      if len(batch) > 0:
         yield self.build_test_batch(batch)
         
   def source_words(self, wordIDs):
      """ maps IDs to source word strings """
      return [self.ID2src[id] for id in wordIDs]

   def target_words(self, wordIDs):
      """ maps IDs to target word strings """
      return [self.ID2tgt[id] for id in wordIDs if id > 0]

