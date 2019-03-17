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
from prediction import predicter
from train import trainIters, trainPrediction
from evaluate import evaluate

MAX_LENGTH  = 1000
SOS_token = 0
EOS_token = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Lang():

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
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




patent_data = np.load('../data/patent_split161.npy')


split = int(.9 * len(patent_data))
train = patent_data[:split]
test = patent_data[split:]


""" -------------------- HELPER FUNCTIONS --------------------  """

def prepare_vocab(data):
    vocab = Lang('corpus')
    for doc in data:
        vocab.addDocument(doc['content'][-MAX_LENGTH+1:])
    return vocab

def prepare_autoencoder_train(p_data):
    """ Prepares the language and data for training on autoencoder network
    @param p_data (list): list of patent documents from file

    returns data: list of documents to train
    returns vocab: language
    """
    data = []
    for doc in p_data:
        data.append(doc['content'][-MAX_LENGTH+1:])
    return data

def prepare_prediction_train(data, encoder, vocab):
    """ Prepares the language and data for training on prediction network
    @param data (list): list of patent documents from file
    @param encoder (nn.Module): encoder model
    @param vocab (Lang): language for training

    returns data: list of documents to train
    returns vocab: language
    """
    patents_present = set()
    for p in data:
        patents_present.add(p['id'])


    pairs = []

    for p in data:
        for c in p['citations']:
            if c in patents_present: pairs.append([(p['id'], c), 1])
    
    all_data = []
    all_data.extend(pairs)

    sample_size = 9 * len(pairs)
    for _ in range(sample_size):
        x = pairs[random.choice(range(len(pairs)))][0][0]
        y = pairs[random.choice(range(len(pairs)))][0][0]
        all_data.append([(x,y), 0])

    docs = dict()

    def findPatent(patent, documents):
        for a in documents:
            if patent == a['id']:
                return a['content'][-MAX_LENGTH+1:]

    for d in all_data:
        patent, citation = d[0]
        if patent not in docs: docs[patent] = findPatent(patent, data)
        if citation not in docs: docs[citation] = findPatent(citation, data)   
        
    all_docs_txt = [] 
    for key in docs:
        if docs[key] is not None:
            all_docs_txt.append(docs[key][-MAX_LENGTH+1:])


    training = []
    

    for pair, target in all_data:

        p,c = pair
 
        p_indexes = [vocab.word2idx[word] for word in docs[p]]
        c_indexes = [vocab.word2idx[word] for word in docs[c]]

        p_indexes.append(EOS_token)
        c_indexes.append(EOS_token)

        input_tensor1 = torch.tensor(p_indexes, dtype=torch.long, device=device).view(-1, 1)
        input_tensor2 = torch.tensor(c_indexes, dtype=torch.long, device=device).view(-1, 1)

        input_length1 = input_tensor1.size(0)

        encoder_hidden1 = encoder.initHidden()

        for ei in range(input_length1-1):
            encoder_output1, encoder_hidden1 = encoder(input_tensor1[ei], encoder_hidden1)


        input_length2 = input_tensor2.size(0)
        
        encoder_hidden2 = encoder.initHidden()

        for ei in range(input_length2):
            encoder_output2, encoder_hidden2 = encoder(input_tensor2[ei], encoder_hidden1)


        training.append((encoder_hidden1, encoder_hidden2, target))


    return docs, all_docs_txt, training

""" ------------------------------------------------------------ """



""" ------------------------ TRAINING --------------------------  """



vocab = prepare_vocab(patent_data)

data = prepare_autoencoder_train(train)

hidden_size = 256
n_words = vocab.n_words
encoder1 = encoder(n_words, hidden_size).to(device)
attn_decoder1 = decoder(hidden_size, n_words).to(device)

predicter1 = predicter(hidden_size, 1).to(device)

trainIters(data, encoder1, attn_decoder1, 100000, vocab)

encoder = torch.load('model/encoder')

docs, docs_txt, training = prepare_prediction_train(train, encoder1, vocab)

trainPrediction(training, predicter1, 100000)

predicter = torch.load('model/predicter')

""" ------------------------------------------------------------ """



""" ------------------------ TESTING ---------------------------  """


evaluate(data, patents, [algos], all_patents, docs, doc_txt, k)

""" ------------------------------------------------------------ """

