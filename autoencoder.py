import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.cell = nn.GRU(hidden_size, hidden_size) # EXTENSION: add LSTM cell


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        return self.cell(embedded, hidden) # output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.cell = nn.GRU(hidden_size, hidden_size) #LSTM??
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed = F.relu(self.embedding(input).view(1, 1, -1))
        output, hidden = self.cell(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
