
import sys
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NNModule(nn.Module):
   """ defines various helper functions which are inherited by neural models """
   
   def __init__(self):
      """ calls the initializer of the parent class """
      super().__init__()

   def on_gpu(self):
      """ checks if the model resides on GPU """
      return next(self.parameters()).is_cuda
   
   def move_to_device(self, x):
      return x.cuda() if self.on_gpu() else x
   
   def long_tensor(self, array):
      return self.move_to_device(torch.LongTensor(array))
   
   def zero_tensor(self, *size):
      return self.move_to_device(torch.zeros(*size))
   
   def one_tensor(self, *size):
      return self.move_to_device(torch.ones(*size))
   
   def zero_long_tensor(self, *size):
      return self.zero_tensor(*size).long()
   

### Encoder ###################################################################

class Encoder(NNModule):
   """ implements the encoder of the encoder-decoder architecture """
   
   def __init__(self, vocab_size, word_emb_size, rnn_size, rnn_depth,
                dropout_rate, emb_dropout_rate, embeddings=None):
      
      super().__init__()

      if embeddings is None:
         # Use randomly initialized word embeddings
         self.embeddings = nn.Embedding(vocab_size, word_emb_size, padding_idx=0)
      else:
         # Use pretrained word embeddings
         self.embeddings = nn.Embedding.from_pretrained(embeddings)

      # Create the (deep) encoder LSTM
      self.rnn = nn.LSTM(word_emb_size, rnn_size, batch_first=True, bidirectional=True)
      self.deep_rnns = nn.ModuleList(
         [nn.LSTM(rnn_size*2, rnn_size, batch_first=True, bidirectional=True)
          for _ in range(rnn_depth-1)])

      # Create separate dropout layers for embeddings and the rest
      self.dropout = nn.Dropout(dropout_rate)
      self.emb_dropout = nn.Dropout(emb_dropout_rate)

      
   def forward(self, wordIDs, seq_len):

      # Look up the source word embeddings
      word_embs = self.embeddings(wordIDs)
      word_embs = self.emb_dropout(word_embs)
      
      # Run the encoder BiRNN
      packed_input = pack_padded_sequence(word_embs, seq_len, batch_first=True)
      output, _ = self.rnn(packed_input)
      states, _ = pad_packed_sequence(output, batch_first=True)
         
      # Run additional deep RNN layers with residual connections (if present)
      for rnn in self.deep_rnns:
         packed_input = pack_padded_sequence(self.dropout(states), seq_len, batch_first=True)
         output, _ = rnn(packed_input)
         output, _ = pad_packed_sequence(output, batch_first=True)
         states = states + output  # residual connections

      return self.dropout(states)


### Attention #################################################################

class Attention(NNModule):

   def __init__(self, enc_rnn_size, dec_rnn_size):
      
      super().__init__()

      self.projection = nn.Linear(enc_rnn_size*2+dec_rnn_size, dec_rnn_size)
      self.final_weights = nn.Parameter(torch.randn(dec_rnn_size))


   def forward(self, enc_states, dec_state, src_len=None):

      # Replicate dec_state along a new (sentence length) dimension
      exp_dec_states = dec_state.unsqueeze(1).expand(-1,enc_states.size(1),-1)

      # Replicate enc_state along the first (batch) dimension if it is 1
      # needed during beam search
      exp_enc_states = enc_states.expand(dec_state.size(0),-1,-1)

      # Append the decoder state to each encoder state
      input = torch.cat((exp_enc_states, exp_dec_states), dim=-1)
      
      # Apply a fully connected layer
      proj_input = torch.tanh(self.projection(input))
      
      # Multiply with the final weight vector to get a single attention score
      # Division by a normalization constant facilitates training
      scores = torch.matmul(proj_input, self.final_weights) / \
               math.sqrt(self.final_weights.size(0))

      if src_len:  # not beam search
         # mask all padding positions
         mask = [[0]*l + [-float('inf')]*(enc_states.size(1)-l) for l in src_len]
         mask = self.move_to_device(torch.Tensor(mask))
         scores = scores + mask
      
      # softmax across all encoder positions
      attn_probs = F.softmax(scores, dim=-1)
      
      # weighted average of encoder representations
      enc_context = torch.sum(enc_states*attn_probs.unsqueeze(2), dim=1)
      
      return enc_context

         
### Decoder ###################################################################

class NMTDecoder(NNModule):

   def __init__(self, src_vocab_size, tgt_vocab_size, word_emb_size,
                enc_rnn_size, dec_rnn_size, enc_depth, dec_depth, 
                dropout_rate, emb_dropout_rate, tie_embeddings=True,
                src_embeddings=None, tgt_embeddings=None):
      ''' intialize the model before training starts '''
      
      super().__init__()

      # We need at least 2 decoder RNNs because the context vector
      # is computed with the hidden state of the first RNN layer.
      dec_depth = min(dec_depth, 2)
      self.dec_rnn_size = dec_rnn_size
      self.dec_depth = dec_depth
      self.apply_projection = (tie_embeddings or tgt_embeddings)

      self.ctx_vec_size = enc_rnn_size*2

      # create the encoder and attention sub-modules
      self.encoder = Encoder(src_vocab_size, word_emb_size, enc_rnn_size, enc_depth,
                             dropout_rate, emb_dropout_rate, src_embeddings)
      self.attention = Attention(enc_rnn_size, dec_rnn_size)

      if tgt_embeddings is None:
         # use randomly initialized target embeddings
         self.tgt_embeddings = nn.Embedding(tgt_vocab_size, word_emb_size)
      else:
         # use pretrained target embeddings
         self.tgt_embeddings = nn.Embedding.from_pretrained(tgt_embeddings)

      # create the (deep) decoder RNN
      self.dec_rnn  = nn.LSTMCell(word_emb_size+self.ctx_vec_size, dec_rnn_size)
      self.deep_dec_rnns = nn.ModuleList(
         [nn.LSTMCell(dec_rnn_size+self.ctx_vec_size, dec_rnn_size)
          for _ in range(1, dec_depth)]
      )

      # allocate separate dropout layers for embeddings and the rest
      self.dropout = nn.Dropout(dropout_rate)
      self.emb_dropout = nn.Dropout(emb_dropout_rate)

      # If we use tied input and output embeddings, we need to
      # project the final hidden state to the target embedding size.
      if self.apply_projection:
         # Create the projection layer
         self.output_proj = nn.Linear(dec_rnn_size+self.ctx_vec_size, word_emb_size)
         # Create the output layer weight matrix
         self.output_layer = nn.Linear(word_emb_size, tgt_vocab_size)
         # Tie input and output embeddings
         self.output_layer.weight = self.tgt_embeddings.weight
      else:
         self.output_layer = nn.Linear(dec_rnn_size+self.ctx_vec_size, tgt_vocab_size)
      

   def finetune_embeddings(self, flag=True):
      self.tgt_embeddings.weight.requires_grad = flag
      self.encoder.embeddings.weight.requires_grad = flag

   
   def init_decoder(self, src_wordIDs, src_len):

      # Run the encoder with the input sequence
      src_wordIDs = self.long_tensor(src_wordIDs)
      enc_states  = self.encoder(src_wordIDs, src_len)

      # Initialize the decoder state
      batch_size = enc_states.size(0)
      init_state = self.zero_tensor(batch_size, self.dec_rnn_size)
      init_state = (init_state, init_state)
      dec_rnn_states = [init_state for _ in range(self.dec_depth)]
      enc_context = self.zero_tensor(batch_size, self.ctx_vec_size)
      
      return enc_states, dec_rnn_states, enc_context
   
   
   def decoder_step(self, prev_word_embs, enc_states, dec_rnn_states,
                    enc_context, src_len=None):
      ''' runs a single decoder step '''

      # Run the first decoder RNN
      input_vectors = torch.cat((prev_word_embs, enc_context), dim=-1)
      dec_rnn_states[0] = self.dec_rnn(input_vectors, dec_rnn_states[0])
      hidden_state = dec_rnn_states[0][0]

      # Compute the source context vector
      enc_context = self.attention(enc_states, hidden_state, src_len)

      # Run additional deep decoder RNN layers with residual connections
      dec_input = hidden_state
      for i, rnn in enumerate(self.deep_dec_rnns):
         # Add context vector to input
         ext_dec_input = self.dropout(torch.cat((dec_input, enc_context), dim=-1))
         # Compute the next decoder layer
         dec_rnn_states[i] = rnn(ext_dec_input, dec_rnn_states[i])
         dec_input = dec_input + dec_rnn_states[i][0]  # residual connections

      return dec_input, dec_rnn_states, enc_context

   
   def compute_scores(self, hidden_states, enc_contexts):
      """ computes the values of the output layer """
      hidden_states = self.dropout(torch.cat((hidden_states, enc_contexts), dim=-1))
      if self.apply_projection:
         hidden_states = self.output_proj(hidden_states)
      return self.output_layer(hidden_states)

   
   def forward(self, src_wordIDs, src_len, tgt_wordIDs):
      ''' forward pass of the network during training and evaluation on dev data '''

      self.train(True)

      enc_states, dec_rnn_states, enc_context = self.init_decoder(src_wordIDs, src_len)
      
      # Look up the target word embeddings
      tgt_word_embs = self.emb_dropout(self.tgt_embeddings(tgt_wordIDs))

      # Run the decoder for each target word and collect the hidden states
      hidden_states = []
      enc_contexts = []
      for i in range(tgt_word_embs.size(1)):
         hidden_state, dec_rnn_states, enc_context = self.decoder_step(
            tgt_word_embs[:,i,:], enc_states, dec_rnn_states, enc_context, src_len)
         hidden_states.append(hidden_state)
         enc_contexts.append(enc_context)

      # Compute the scores of the output layer
      hidden_states = torch.stack(hidden_states, dim=1)
      enc_contexts = torch.stack(enc_contexts, dim=1)
      scores = self.compute_scores(hidden_states, enc_contexts)

      return scores


   ### Translation ########################

   def translate(self, src_wordIDs, src_len, beam_size=0):
      ''' forward pass of the network during translation '''

      self.train(False)

      if beam_size > 0:
         return self.beam_translate(src_wordIDs, src_len, beam_size)

      # run the encoder and initialize the decoder states
      enc_states, dec_rnn_states, enc_context = self.init_decoder(src_wordIDs, src_len)

      tgt_wordIDs = []
      prev_wordIDs = self.zero_long_tensor(len(src_wordIDs))
      tgt_logprobs = self.zero_tensor(len(src_wordIDs))
      nonfinal     = self.one_tensor(len(src_wordIDs))
      
      # target sentences may have twice the size of the source sentences plus 5
      for i in range(src_len[0]*2+5):

         # run the decoder RNN for a single step
         hidden_state, dec_rnn_states, enc_context = self.decoder_step(
            self.tgt_embeddings(prev_wordIDs), enc_states, dec_rnn_states,
            enc_context, src_len)
         scores = self.compute_scores(hidden_state, enc_context)

         # extract the most likely target word for each sentence
         best_logprobs, best_wordIDs = F.log_softmax(scores, dim=-1).max(dim=-1)
         
         tgt_wordIDs.append(best_wordIDs)
         prev_wordIDs = best_wordIDs

         # sum up log probabilities until the end symbol with index 0 is encountered
         tgt_logprobs += best_logprobs * nonfinal
         nonfinal *= (best_wordIDs != 0).float()
         
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      tgt_wordIDs = torch.stack(tgt_wordIDs).t().cpu().data.numpy()
      return tgt_wordIDs, tgt_logprobs


   ### Translation with beam decoding ########################

   def build_beam(self, logprobs, beam_size, dec_rnn_states):

      # get the threshold which needs to be exceeded by all hypotheses in the new beam
      top_logprobs, _ = logprobs.view(-1).topk(beam_size+1)
      threshold = top_logprobs[-1]

      # extract the most likely extensions for each hypothesis
      top_logprobs, top_wordIDs = logprobs.topk(beam_size, dim=-1)

      # extract the most likely extended hypotheses overall
      new_wordIDs = []
      new_logprobs = []
      prev_pos = []
      for i in range(top_logprobs.size(0)):
         for k in range(top_logprobs.size(1)):
            if (top_logprobs[i,k] <= threshold).all(): # without all() it doesn't work
               break # ignore the rest
            prev_pos.append(i)
            new_wordIDs.append(top_wordIDs[i,k])
            new_logprobs.append(top_logprobs[i,k])
      new_wordIDs = torch.stack(new_wordIDs)
      new_logprobs = torch.stack(new_logprobs)
      new_dec_states = []
      for d in range(self.dec_depth):
         hidden_states = torch.stack([dec_rnn_states[d][0][i] for i in prev_pos])
         cell_states   = torch.stack([dec_rnn_states[d][1][i] for i in prev_pos])
         new_dec_states.append((hidden_states, cell_states))

      return new_wordIDs, new_logprobs, new_dec_states, prev_pos

   
   def beam_translate(self, src_wordIDs, src_len, beam_size):
      ''' processes a single sentence with beam search '''
      
      enc_states, dec_rnn_states, enc_context = self.init_decoder(src_wordIDs, src_len)
      
      tgt_wordIDs = []
      prev_pos = []
      prev_wordIDs = self.zero_long_tensor(1)
      prev_logprobs = self.zero_tensor(1)
      
      # target sentences have at most twice the size of the source sentences plus 5
      for i in range(src_len[0]*2+5):

         # compute scores for the next target word candidates
         tgt_word_embs = self.tgt_embeddings(prev_wordIDs)
         hidden_state, dec_rnn_states, enc_context = self.decoder_step(
            tgt_word_embs, enc_states, dec_rnn_states, enc_context, src_len)
         scores = self.compute_scores(hidden_state, enc_context)

         # add the current logprob to the logprob of the previous hypothesis
         logprobs = prev_logprobs.unsqueeze(1) + F.log_softmax(scores, dim=-1)

         # extract the best hypotheses
         best_wordIDs, prev_logprobs, dec_rnn_states, prev \
            = self.build_beam(logprobs, beam_size, dec_rnn_states)

         # store information for computing the best translation at the end
         tgt_wordIDs.append(best_wordIDs.cpu().data.numpy().tolist())
         prev_pos.append(prev)
         prev_wordIDs = best_wordIDs
         
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      # extract the best translation
      # get the position of the most probable hypothesis
      logprob, pos = prev_logprobs.max(-1)
      pos = int(pos)

      # extract the best translation backward using prev_pos
      wordIDs = []
      for i in range(len(prev_pos)-1,0,-1):
         pos = prev_pos[i][pos]
         wordIDs.append(tgt_wordIDs[i-1][pos])
      wordIDs.reverse()

      return [wordIDs], [logprob]
