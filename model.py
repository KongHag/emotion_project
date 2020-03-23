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
import torch.nn.functional as F


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
    _features_idx = {
        "acc": [0, 256],
        "cedd": [256, 400],
        "cl": [400, 433],
        "eh": [433, 513],
        "fcth": [513, 705],
        "gabor": [705, 765],
        "jcd": [765, 933],
        "sc": [933, 997],
        "tamura": [997, 1015],
        "lbp": [1015, 1271],
        "fc6": [1271, 5367]
    }

    def __init__(self, dropout):
        super(RecurrentNetFeature, self).__init__()

        self.conv1d_acc_1 = nn.Conv1d(
            in_channels=4, out_channels=8, kernel_size=5)
        self.maxpool1d_acc_1 = nn.MaxPool1d(
            kernel_size=2)
        self.conv1d_acc_2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=5)
        self.maxpool1d_acc_2 = nn.MaxPool1d(
            kernel_size=2)

        self.conv1d_cedd_1 = nn.Conv1d(
            in_channels=6, out_channels=12, kernel_size=5)
        self.maxpool1d_cedd_1 = nn.MaxPool1d(
            kernel_size=2)
        self.conv1d_cedd_2 = nn.Conv1d(
            in_channels=12, out_channels=24, kernel_size=5)
        self.maxpool1d_cedd_2 = nn.MaxPool1d(
            kernel_size=2)

        self.conv2d_eh_1 = nn.Conv2d(
            in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.maxpool2d_eh_1 = nn.MaxPool2d(
            kernel_size=2)
        
        self.conv1d_fcth_1 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=5)
        self.maxpool1d_fcth_1 = nn.MaxPool1d(
            kernel_size=2)
        self.conv1d_fcth_2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1d_fcth_2 = nn.MaxPool1d(
            kernel_size=2)

        self.conv1d_jcd_1 = nn.Conv1d(
            in_channels=7, out_channels=14, kernel_size=5, padding=2)
        self.maxpool1d_jcd_1 = nn.MaxPool1d(
            kernel_size=2)
        self.conv1d_jcd_2 = nn.Conv1d(
            in_channels=14, out_channels=28, kernel_size=3, padding=1)
        self.maxpool1d_jcd_2 = nn.MaxPool1d(
            kernel_size=2)

        self.cnn_out_dim = 16*13 + 24*3 + 10*2*2 + 32*5 + 28*6 + 4096

        self.lstm_layer = nn.LSTM(input_size=self.cnn_out_dim,
                                  hidden_size=self.cnn_out_dim,
                                  num_layers=1,
                                  dropout=dropout,
                                  bias=True,
                                  batch_first=True,
                                  bidirectional=True)

        self.dropout1 = nn.Dropout(dropout)

        self.fc_layer = nn.Linear(
            in_features=self.cnn_out_dim*2, out_features=2, bias=True)

    def forward(self, X, hidden_and_cell):

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        X_acc = X[:, :, self._features_idx["acc"]
                  [0]:self._features_idx["acc"][1]]
        X_acc = torch.reshape(X_acc, (batch_size * seq_len, 4, 64))
        X_acc = F.relu(self.conv1d_acc_1(X_acc))
        X_acc = self.maxpool1d_acc_1(X_acc)
        X_acc = F.relu(self.conv1d_acc_2(X_acc))
        X_acc = self.maxpool1d_acc_2(X_acc)
        X_acc = torch.reshape(X_acc, (X_acc.shape[0], 16*13))

        X_cedd = X[:, :, self._features_idx["cedd"]
                   [0]:self._features_idx["cedd"][1]]
        X_cedd = torch.reshape(X_cedd, (batch_size * seq_len, 6, 24))
        X_cedd = F.relu(self.conv1d_cedd_1(X_cedd))
        X_cedd = self.maxpool1d_cedd_1(X_cedd)
        X_cedd = F.relu(self.conv1d_cedd_2(X_cedd))
        X_cedd = self.maxpool1d_cedd_2(X_cedd)
        X_cedd = torch.reshape(X_cedd, (X_cedd.shape[0], 24*3))

        X_eh = X[:, :, self._features_idx["eh"]
                 [0]:self._features_idx["eh"][1]]
        X_eh = torch.reshape(X_eh, (batch_size * seq_len, 5, 4, 4))
        X_eh = F.relu(self.conv2d_eh_1(X_eh))
        X_eh = self.maxpool2d_eh_1(X_eh)
        X_eh = torch.reshape(X_eh, (X_eh.shape[0], 10*2*2))

        X_fcth = X[:, :, self._features_idx["fcth"]
                   [0]:self._features_idx["fcth"][1]]
        X_fcth = torch.reshape(X_fcth, (batch_size * seq_len, 8, 24))
        X_fcth = F.relu(self.conv1d_fcth_1(X_fcth))
        X_fcth = self.maxpool1d_fcth_1(X_fcth)
        X_fcth = F.relu(self.conv1d_fcth_2(X_fcth))
        X_fcth = self.maxpool1d_fcth_2(X_fcth)
        X_fcth = torch.reshape(X_fcth, (X_fcth.shape[0], 32*5))

        X_jcd = X[:, :, self._features_idx["jcd"]
                   [0]:self._features_idx["jcd"][1]]
        X_jcd = torch.reshape(X_jcd, (batch_size * seq_len, 7, 24))
        X_jcd = F.relu(self.conv1d_jcd_1(X_jcd))
        X_jcd = self.maxpool1d_jcd_1(X_jcd)
        X_jcd = F.relu(self.conv1d_jcd_2(X_jcd))
        X_jcd = self.maxpool1d_jcd_2(X_jcd)
        X_jcd = torch.reshape(X_jcd, (X_jcd.shape[0], 28*6))

        X_fc6 = X[:, :, self._features_idx["fc6"]
                   [0]:self._features_idx["fc6"][1]]
        X_fc6 = torch.reshape(X_fc6, (batch_size * seq_len, 4096))

        X = torch.cat((X_acc, X_cedd, X_eh, X_fcth, X_jcd, X_fc6), dim=1)
        X = torch.reshape(X, (batch_size, seq_len, self.cnn_out_dim))

        # fc_outputs = (output_acc, output_cedd)
        # X = torch.cat(fc_outputs, dim=2)
        X, hidden = self.lstm_layer(X, hidden_and_cell)
        X = F.relu(X)
        X = self.dropout1(X)
        X = F.relu(self.fc_layer(X))
        return X

    def initHelper(self, batch_size):
        # initialize hidden states to 0
        hidden = Variable(torch.zeros(
            2, batch_size, self.cnn_out_dim))
        cell = Variable(torch.zeros(
            2, batch_size, self.cnn_out_dim))

        return hidden, cell


if __name__ == '__main__':
    from dataset import MediaEval18
    from torch.utils.data import DataLoader

    my_set = MediaEval18(root='./data', seq_len=20,
                         fragment=0.1, features=['visual'])
    my_loader = DataLoader(my_set, batch_size=32)

    X, Y = next(iter(my_loader))

    model = RecurrentNetFeature(dropout=0.3)

    print(model)
    hidden_and_cell = model.initHelper(batch_size=32)
    output = model(X, hidden_and_cell)
    print(output.shape)
