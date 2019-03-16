#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
run.py: Run file
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
import random
from autoencoder import encoder, decoder
from train import trainIters

MAX_LENGTH  = 30000
SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
       """ Init Language.

        @param name (string): the name of the language
        """
        self.name = name
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addDocument(self, document):
        """ Adds all words in document to the language

        @param document (list): list of words
        """
        for word in document:
            self.addWord(word)

    def addWord(self, word):
        """ Adds a single word to the language

        @param document (string): word to add to language
        """
        if word not in self.word2index:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




patent_data = np.load('../data/patent_fuller.npy')

def prepare_train(p_data):
     """ Prepares the language and data for training
    @param p_data (list): list of patent documents from file

    returns data: list of documents to train
    returns vocab: language
    """
    vocab = Lang('corpus')
    data = []
    for doc in p_data:
        vocab.addDocument(doc['content'][-100:])
        data.append(doc['content'][-100:])
    return data, vocab


data, vocab = prepare_train(patent_data)

hidden_size = 256
n_words = vocab.n_words
encoder1 = encoder(n_words, hidden_size).to(device)
attn_decoder1 = decoder(hidden_size, n_words).to(device)


trainIters(data, encoder1, attn_decoder1, 1000, vocab)





