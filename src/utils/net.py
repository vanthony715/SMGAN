# -*- coding: utf-8 -*-
"""
@author: avasque1@jh.edu
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Generator(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, max_length: int, num_heads: int, dropout_prob: float,
                bidirectional: bool) -> None:
        '''
        Bidirectional Recurrent Generator with Attention Layer
        '''
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  #generate embeddings
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_prob, batch_first=True, bidirectional=bidirectional) #gru instead of lstm
        if bidirectional: #if bidirectional, multiply by 2
            self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout_prob)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_prob)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_prob) #very prone to overfitting, so dropout or some type of regularization is needed
        self.max_length = max_length

    def forward(self, z: torch.tensor) -> torch.tensor:
        '''
        Builds graph and forwards
        '''
        embedded = self.embedding(z)  # Shape: (batch_size, max_length, embedding_dim)
        gru_out, _ = self.gru(embedded)  # Shape: (batch_size, max_length, hidden_dim)
        gru_out = self.dropout(gru_out)
        
        #multi-head attention (MultiheadAttention expects (seq_len, batch_size, hidden_dim))
        attn_input = gru_out.transpose(0, 1)  #Transpose -> (max_length, batch_size, hidden_dim)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)  # Self-attention
        attn_out = attn_out.transpose(0, 1)  # Transpose back -> (batch_size, max_length, hidden_dim)

        output = self.fc(attn_out)  #output layer
        output_indices = torch.argmax(output, dim=-1)  #Convert from continous to discrete indices for Discriminator emvedding layer
        return output_indices 

class Discriminator(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, max_length: int, num_heads: int, dropout_prob: float,
                bidirectional: bool) -> None:
        '''
        Bidirectional Recurrent Discriminator with Attention Layer
        '''
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, dropout=dropout_prob, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        ##if bidirectional, then reduce the hidden dimensionality using fully connected of hid_size/2
        if bidirectional:
            self.fc_reduce = nn.Linear(hidden_dim * 2, hidden_dim)  #reduce dim

        ##attend
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.max_length = max_length

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Builds graph and forwards
        '''
        embedded = self.embedding(x)  #shape: (batch_size, max_length, embedding_dim)
        gru_out, _ = self.gru(embedded)  #shape: (batch_size, max_length, hidden_dim)
        gru_out = self.fc_reduce(gru_out)  #reduce dimension -> hidden_dim
        gru_out = self.dropout(gru_out)

        ##ulti-head attention
        attn_input = gru_out.transpose(0, 1)  #transpose -> (max_length, batch_size, hidden_dim)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)  #self-attention
        attn_out = attn_out.transpose(0, 1)  #transpose -> (batch_size, max_length, hidden_dim)

        ##take the last output from the sequence (many-to-one)
        output = self.fc(attn_out[:, -1, :])  ##hape: (batch_size, 1)
        return torch.sigmoid(output)  ##igmoid for binary classification
