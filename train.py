#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
train.py: Train File
John Knowles <jkn0wles@stanfordedu>
Sam Premutico <samprem@stanford.edu>
"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


MAX_LENGTH  = 100
SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
""" -------------------- HELPER FUNCTIONS --------------------  """

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    print(since, percent)
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

""" ------------------------------------------------------------ """



def indexesFromDocument(lang, document):
    """ Gets the index of each word in the document.
   
    Code edited from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    @param lang (Lang): the language from the corpus
    @param document (list): list of words in a patent 

    returns: list of indices (integers) of words in document
    """
    return [lang.word2idx[word] for word in document]


def tensorFromDocument(lang, document):
    """ Creates a tensor from a list of integers
    
    Code edited from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    @param lang (Lang): the language from the corpus
    @param document (list): list of words in a patent 

    returns: torch tensor of integers
    """
    indexes = indexesFromDocument(lang, document)
    indexes.append(1)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromInput(pair, lang):
    """ Computes input, target tensors for training

    Code edited from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    @param pair (list): input, target pair of documents 
    @param lang (Lang): the language from the corpus
   
    returns: (input tensor, target tensor)
    """
    input_tensor = tensorFromDocument(lang, pair)
    return (input_tensor, input_tensor)


def train(input_tensor, target_tensor, encoder, decoder, e_optimizer, d_optimizer, criterion, max_length=MAX_LENGTH, verbose=False):
    """ Trains a forward pass over autoencoder network

    Code from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    @param input_tensor (Tensor): tensor of the input document
    @param target_tensor (Tensor): tensor of the target document
    @param encoder (nn.Module): encoder model 
    @param decoder (nn.Module): decoder model 
    @param e_optimizer (nn.optim): encoder optimizer
    @param d_optimizer (nn.optim): decoder optimizer
    @param criterion (nn.Module): loss function

    returns: loss of forward pass through network
    """    
    encoder_hidden = encoder.initHidden()

    e_optimizer.zero_grad()
    d_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    if (verbose): print(input_length)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    if (verbose): print('encoded')
    decoder_input = torch.tensor([[0]], device=device)

    decoder_hidden = encoder_hidden
    use_teacher_forcing = True
    #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    if (verbose): print('decoded')

    loss.backward()
    if (verbose): print('loss')
    e_optimizer.step()
    d_optimizer.step()
    if (verbose): print('optimize')
    return loss.item() / target_length


def trainIters(data, encoder, decoder, n_iters, lang,  print_every=100, plot_every=100, learning_rate=0.01):
    """ Trains a encoder-decoder network over n_iters

    Code from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    @param data (list): list of list of strings (documents)
    @param encoder (nn.Module): encoder model 
    @param decoder (nn.Module): decoder model 
    @param n_iters (int): number of iterations
    @param lang (Lang): language used 

    """   
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromInput(random.choice(data), lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        #print(iter)

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(iter)


        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # save the encoder model
    torch.save(encoder, 'model/encoder2')
