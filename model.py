# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:46:42 2020

@author: Tim
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
# %%


class RecurrentNet(nn.Module):
    """
    """

    def __init__(self, in_dim, hid_dim, num_hid, out_dim, dropout=0, bidirectional=False):
        super(RecurrentNet, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_hid = num_hid
        self.dropout = dropout

        self.lstm_layer = nn.LSTM(input_size=self.in_dim,
                                  hidden_size=self.hid_dim,
                                  num_layers=self.num_hid,
                                  batch_first=True,
                                  dropout=self.dropout,
                                  bias=True,
                                  bidirectional=bidirectional)

        self.out_layer = nn.Linear(self.hid_dim, out_dim, bias=True)

    def forward(self, in_x, hidden):
        output, hidden = self.lstm_layer(in_x, hidden)

        output = self.out_layer(output)

        return output

    def initHelper(self, batch_size):
        # initialize hidden states to 0
        hidden = Variable(torch.zeros(self.num_hid, batch_size, self.hid_dim))
        cell = Variable(torch.zeros(self.num_hid, batch_size, self.hid_dim))

        return hidden, cell

import torch.nn.functional as F

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(5367, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x