#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
autoencoder.py: AutoEncoder Model
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class encoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		 """ Init Encoder.

        @param input_size (int): size of the vocab (for embedding)
        @param hidden_size (int): hidden size of the rnn cell
        """
		super(encoder, self).__init__()
		self.hidden_size = hidden_size
    	self.embedding = nn.Embedding(input_size, hidden_size)
    	self.cell = nn.GRU(hidden_size, hidden_size) # EXTENSION: add LSTM cell


    def forward(self, input, hidden):
    	 """ Forward pass of encoder

        @param input: tensor of integers, shape (c, batch)
        @param hidden: internal state of the RNN before reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns output: 
        @returns hidden: internal state of the RNN after reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)
        """
        print("INPUT", input.size, "HIDDEN", hidden.size)
    	embedded = self.embedding(input).view(1, 1, -1)
    	output, hidden = self.cell(output, hidden)
    	print("output", output.size, "HIDDEN", hidden.size)
    	return self.cell(output, hidden) # output, hidden

    def initHidden(self):
    	""" Initialize hidden state of RNN

        @returns: tensor of zeros, shape (1, 1, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)


class decoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		""" Init Decoder.

        @param hidden_size (int): hidden size of the rnn cell
        @param output_size (int): size of the vocab (for embedding)

        """
		super(decoder, self).__init__()
		self.embedding = nn.Embedding(output_size, hidden_size)
        self.cell = nn.GRU(hidden_size, hidden_size) #LSTM??
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
    	 """ Forward pass of decoder

        @param input: tensor of integers,
         shape (c, batch)
        @param hidden: internal state of the RNN before reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns output: 
        @returns hidden: internal state of the RNN after reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)
        """
       	print("INPUT", input.size, "HIDDEN", hidden.size)
    	embed = F.relu(self.embedding(input).view(1, 1, -1))
    	output, hidden = self.cell(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
    	""" Initialize hidden state of RNN

        @returns: tensor of zeros, shape (1, 1, hidden_size)
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

class autoencoder(nn.Module):
	def __init__(self, encoder, decoder, device):
		""" Init Autoencoder.

        @param encoder: encoder module
        @param decoder: decoder module
        @param device (string): device in use
        """
		super(autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
