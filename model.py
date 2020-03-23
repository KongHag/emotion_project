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

    def __init__(self):
        super(RecurrentNetFeature, self).__init__()

        self.conv1d_acc_1 = torch.nn.Conv1d(in_channels=4, out_channels=8,
                                            kernel_size=5, stride=3)
        self.conv1d_acc_2 = torch.nn.Conv1d(in_channels=8, out_channels=16,
                                            kernel_size=5, stride=3)
        self.fc_layers_acc = nn.Linear(
            in_features=16*6, out_features=2, bias=True)
        # self.fc_layers_cedd = nn.Linear(
        #     len(self._features_len["cedd"]), 32, bias=True)

        # self.lstm_layer = nn.LSTM(input_size=32*2,
        #                           hidden_size=self.hidden_size,
        #                           num_layers=1,
        #                           bias=True,
        #                           batch_first=True,
        #                           bidirectional=True)

        # self.out_layer = nn.Linear(2*self.hidden_size, 2, bias=True)

    def forward(self, X):

        X_acc = X[:, :, :len(self._features_len["acc"])]
        batch_size = X_acc.shape[0]
        seq_len = X_acc.shape[1]
        X_acc = torch.reshape(X_acc, (batch_size * seq_len, 4, 64))

        X_acc = self.conv1d_acc_1(X_acc)
        X_acc = self.conv1d_acc_2(X_acc)
        X_acc = torch.reshape(X_acc, (X_acc.shape[0], 16*6))
        X_acc = self.fc_layers_acc(X_acc)
        X_acc = torch.reshape(X_acc, (batch_size, seq_len, 2))

        # X_cedd = X[:, :, :len(self._features_len["cedd"])]
        # output_cedd = self.fc_layers_cedd(X_cedd)

        # fc_outputs = (output_acc, output_cedd)
        # X = torch.cat(fc_outputs, dim=2)
        # X, hidden = self.lstm_layer(X, hidden_cell)
        # X = self.out_layer(X)
        return X_acc

    # def initHelper(self, batch_size):
    #     # initialize hidden states to 0
    #     hidden = Variable(torch.zeros(
    #         2, batch_size, self.hidden_size))
    #     cell = Variable(torch.zeros(
    #         2, batch_size, self.hidden_size))

    #     return hidden, cell


if __name__ == '__main__':
    from dataset import MediaEval18
    from torch.utils.data import DataLoader

    my_set = MediaEval18(root='./data', seq_len=20,
                         fragment=0.001, features=['visual'])
    my_loader = DataLoader(my_set, batch_size=16)

    X, Y = next(iter(my_loader))

    model = RecurrentNetFeature()

    print(model)

    output = model(X)
    print(output.shape)
