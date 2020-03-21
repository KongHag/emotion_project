# -*- coding: utf-8 -*-
"""
TODO : write a doctring
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout, bidirectional):
        super(RecurrentNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.coef = 2 if self.bidirectional else 1

        self.lstm_layer = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  bias=True,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=self.bidirectional)
        self.out_layer = nn.Linear(self.coef*self.hidden_size, output_size, bias=True)

    def forward(self, X, hidden):
        X, hidden = self.lstm_layer(X, hidden)
        X = self.out_layer(X)

        return X

    def initHelper(self, batch_size):
        # initialize hidden states to 0
        hidden = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))

        return hidden, cell
