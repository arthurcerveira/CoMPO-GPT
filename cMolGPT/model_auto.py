import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
########################
# x1 -> y
#######################
class ConditionalTransformer(nn.Module):
    def __init__(self, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1, args = None,
                 emb_input_size=7):
        super(ConditionalTransformer, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=args.nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        tgt_vocab_size = src_vocab_size        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.emb = nn.Embedding(emb_input_size, dim_feedforward, padding_idx=0)
        self.condition_proj = nn.Linear(dim_feedforward, emb_size)

        self.params = nn.ModuleDict({
            # 'conditional': nn.ModuleList([self.emb]),
            'conditional': nn.ModuleList([self.emb, self.condition_proj]),
            'generation': nn.ModuleList([
                self.transformer_decoder, self.positional_encoding, self.tgt_tok_emb, self.generator
            ])
        })

    def forward(self, trg: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor, condition: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        s, b = trg.size()

        # memory = self.emb(condition).unsqueeze(0).repeat(s, 1, 1)
        # outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
        #                                 tgt_padding_mask)
        
        # Generate and project condition embedding to match embedding size
        condition_emb = self.condition_proj(self.emb(condition)).unsqueeze(0).repeat(s, 1, 1)

        # Combine embeddings
        combined_emb = tgt_emb + condition_emb  # Summing instead of concatenating as an alternative

        outs = self.transformer_decoder(combined_emb, condition_emb, tgt_mask, None, tgt_padding_mask)
        return self.generator(outs)

    def decode(self, tgt: Tensor, tgt_mask: Tensor, condition: Tensor):
        s, b = tgt.size()
        # memory = self.emb(condition).unsqueeze(0).repeat(s, 1, 1)

        # return self.transformer_decoder(self.positional_encoding(
        #                   self.tgt_tok_emb(tgt)), memory,
        #                   tgt_mask)
            # Generate condition embedding
        condition_emb = self.emb(condition).unsqueeze(0).repeat(s, 1, 1)
        
        # Project condition embedding if needed (only if used in forward method)
        condition_emb = self.condition_proj(condition_emb)
        
        # Positional encoding of target tokens
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        # Combine condition embedding with target token embedding
        combined_emb = tgt_emb + condition_emb  # Can use concatenation if desired
        
        # Decode with the modified embedding
        return self.transformer_decoder(combined_emb, condition_emb, tgt_mask)

    def decode_exclude(self, tgt: Tensor, tgt_mask: Tensor, target: Tensor, exclude_target: Tensor):
        s, b = tgt.size()

        memory = self.emb(target).unsqueeze(0).repeat(s, 1, 1)
        exclude_memory = self.emb(exclude_target).unsqueeze(0).repeat(s, 1, 1)

        memory_diff = memory - exclude_memory
        # memory_diff = self.ffn(memory_diff)

        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory_diff,
                          tgt_mask)

    def decode_multitarget(self, tgt: Tensor, tgt_mask: Tensor, target: Tensor, aggregate_fn='mean'):
        s, b = tgt.size()
        memory = self.emb(target).unsqueeze(0).repeat(s, 1, 1)

        if aggregate_fn == 'mean':
            pooled_memory = memory.mean(dim=1).unsqueeze(1)  # .repeat(1, b, 1)
        elif aggregate_fn == 'max':
            pooled_memory = memory.max(dim=1).values.unsqueeze(1)
        elif aggregate_fn == 'sum':
            pooled_memory = memory.sum(dim=1).unsqueeze(1)
        else:
            raise ValueError(f"Invalid aggregate_fn: {aggregate_fn}")

        # pooled_memory = self.ffn(pooled_memory)
        # Project condition embedding if needed (only if used in forward method)
        condition_emb = self.condition_proj(pooled_memory)
        
        # Positional encoding of target tokens
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        # Combine condition embedding with target token embedding
        combined_emb = tgt_emb + condition_emb  # Can use concatenation if desired
        
        # Decode with the modified embedding
        return self.transformer_decoder(combined_emb, condition_emb, tgt_mask)

        # return self.transformer_decoder(self.positional_encoding(
        #                   self.tgt_tok_emb(tgt)), pooled_memory,
        #                   tgt_mask)


######################################################################
# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
# 

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


######################################################################
# We create a ``subsequent word`` mask to stop a target word from
# attending to its subsequent words. We also create masks, for masking
# source and target padding tokens
# 

def generate_square_subsequent_mask(sz, DEVICE='cuda'):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt, DEVICE='cuda'):
  tgt_seq_len = tgt.shape[0]
  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return tgt_mask, tgt_padding_mask




