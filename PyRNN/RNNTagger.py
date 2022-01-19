
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class NNModule(nn.Module):
   ''' extends nn.Module with some auxiliary functions '''

   def __init__(self):
      super(NNModule, self).__init__()

   def on_gpu(self):
      ''' check if the model resides on GPU '''
      return next(self.parameters()).is_cuda

   def variable(self, x):
      return Variable(x).cuda() if self.on_gpu() else Variable(x)

   def long_tensor(self, x):
      return self.variable(torch.LongTensor(x))

   def float_tensor(self, x):
      return self.variable(torch.Tensor(x))
   

class WordRepresentation(NNModule):
   '''
   Deep RNN with residual connections for computing 
   character-based word representations
   '''
   def __init__(self, num_chars, emb_size, rec_size, rnn_depth, dropout_rate):
      super(WordRepresentation, self).__init__()

      # character embedding lookup table
      self.embeddings = nn.Embedding(num_chars, emb_size)

      # character-based LSTMs
      self.fwd_rnn = nn.LSTM(emb_size, rec_size)
      self.bwd_rnn = nn.LSTM(emb_size, rec_size)

      # additional RNN layers
      self.deep_rnns = nn.ModuleList([])
      for _ in range(rnn_depth-1):
         self.deep_rnns.append(nn.LSTM(2*rec_size, rec_size, bidirectional=True))

      self.dropout = nn.Dropout(dropout_rate)
      
         
   def forward(self, fwd_charIDs, bwd_charIDs):
      # swap the 2 dimensions and lookup the embeddings
      fwd_embs = self.embeddings(fwd_charIDs.t())
      bwd_embs = self.embeddings(bwd_charIDs.t())

      # run the biLSTM over characters
      fwd_outputs, _ = self.fwd_rnn(fwd_embs)
      bwd_outputs, _ = self.bwd_rnn(bwd_embs)

      # concatenate the forward and backward final states to form
      # word representations
      word_reprs = torch.cat((fwd_outputs[-1], bwd_outputs[-1]), -1)

      # additional RNN layers with residual connections
      for rnn in self.deep_rnns:
         outputs, _ = rnn(self.dropout(word_reprs.unsqueeze(0)))
         word_reprs = word_reprs + outputs.squeeze(0)
      
      return word_reprs


class ResidualLSTM(NNModule):
   ''' Deep BiRNN with residual connections '''
   
   def __init__(self, input_size, rec_size, num_rnns, dropout_rate):
      super(ResidualLSTM, self).__init__()
      self.rnn = nn.LSTM(input_size, rec_size, 
                         bidirectional=True, batch_first=True)

      self.deep_rnns = nn.ModuleList([
         nn.LSTM(2*rec_size, rec_size, bidirectional=True, batch_first=True)
         for _ in range(num_rnns-1)])
      
      self.dropout = nn.Dropout(dropout_rate)

   def forward(self, state):
      state, _ = self.rnn(state)
      for rnn in self.deep_rnns:
            hidden, _ = rnn(self.dropout(state))
            state = state + hidden # residual connection
      return state


class RNNTagger(NNModule):
   ''' main tagger module '''

   def __init__(self, num_chars, num_tags, char_emb_size, word_emb_size,
                char_rec_size, word_rec_size, char_rnn_depth, word_rnn_depth,
                dropout_rate):

      super(RNNTagger, self).__init__()

      # character-based BiLSTMs
      self.word_representations = \
          WordRepresentation(num_chars, char_emb_size, char_rec_size,
                             char_rnn_depth, dropout_rate)
      # word-based BiLSTM
      self.word_rnn = ResidualLSTM(word_emb_size+2*char_rec_size,
                                   word_rec_size, word_rnn_depth, dropout_rate)
      # output MLP net
      self.output_layer = nn.Linear(2*word_rec_size, num_tags)

      # dropout layers
      self.dropout = nn.Dropout(dropout_rate)


   def forward(self, fwd_charIDs, bwd_charIDs, word_embs=None):

      # compute the character-based word representations
      word_reprs = self.word_representations(fwd_charIDs, bwd_charIDs)

      # append word embeddings if available
      if word_embs is not None:
         word_reprs = torch.cat((word_reprs, word_embs), -1)

      # apply dropout
      word_reprs = self.dropout(word_reprs)
         
      # run the BiLSTM over words
      reprs = self.word_rnn(word_reprs.unsqueeze(0)).squeeze(0)
      reprs = self.dropout(reprs)  # and apply dropout
      
      # apply the output layers
      scores = self.output_layer(reprs)

      return scores
