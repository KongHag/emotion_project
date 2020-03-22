# -*- coding: utf-8 -*-
"""
Defines the recurrent network.
The network used is a LSTM network
Input >> LSTM layer >> FC layer >> Output

To setup recurrent net :
>>> from model import RecurrentNet
>>> model = RecurrentNet(input_size, hidden_size, num_layers, output_size, dropout, bidirectional)
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
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(
            self.coef*self.hidden_size, output_size, bias=True)

    def forward(self, X, hidden):
        X, hidden = self.lstm_layer(X, hidden)
        X = self.dropout_layer(X)
        X = self.out_layer(X)

        return X

    def initHelper(self, batch_size):
        # initialize hidden states to 0
        hidden = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))

        return hidden, cell


class RecurrentNetFeature(nn.Module):
    _features_len = {
        "acc": range(0, 256),
        "cedd": range(256, 400),
        "cl": range(400, 433),
        "eh": range(433, 513),
        "fcth": range(513, 705),
        "gabor": range(705, 765),
        "jcd": range(765, 933),
        "sc": range(933, 997),
        "tamura": range(997, 1015),
        "lbp": range(1015, 1271),
        "fc6": range(1271, 5367)
    }

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout, bidirectional):
        super(RecurrentNetFeature, self).__init__()

        self.hidden_size = hidden_size

        self.fc_layers = [
            (feature_idx,
            nn.Linear(len(feature_idx), 32, bias=True))
            for feature_idx in self._features_len.values()
        ]

        self.lstm_layer = nn.LSTM(input_size=32*len(self.fc_layers),
                                  hidden_size=hidden_size,
                                  num_layers=1,
                                  bias=True,
                                  batch_first=True,
                                  bidirectional=True)

        self.out_layer = nn.Linear(2*self.hidden_size, 2, bias=True)

    def forward(self, X, hidden_cell):
        Xs = []
        for feature_idx, fc_layer in self.fc_layers:
            pX = X[:,:,feature_idx]
            output = fc_layer(pX)
            Xs.append(output)
        X = torch.cat(Xs,dim=2)
        X, hidden = self.lstm_layer(X, hidden_cell)
        X = self.out_layer(X)
        return X

    def initHelper(self, batch_size):
        # initialize hidden states to 0
        hidden = Variable(torch.zeros(
            2, batch_size, self.hidden_size))
        cell   = Variable(torch.zeros(
            2, batch_size, self.hidden_size))

        return hidden, cell


if __name__ == '__main__':
    from dataset import MediaEval18
    from torch.utils.data import DataLoader

    my_set = MediaEval18(root='./data', seq_len=20, fragment=0.001, features=['visual'])
    my_loader = DataLoader(my_set, batch_size=16)

    X, Y = next(iter(my_loader))

    model = RecurrentNetFeature(hidden_size = 32)

    hidden, cell = model.initHelper(X.shape[0])
    output = model(X, (hidden, cell))


