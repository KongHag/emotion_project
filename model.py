# -*- coding: utf-8 -*-
"""
Defines three networks.

FCNet               : Input >> FC layer >> Output
RecurrentNet        : Input >> LSTM layer >> FC layer >> Output
RecurrentNetWithCNN : Input >> CNN >> LSTM layer >> FC layer >> Output

To setup recurrent net :
>>> from model import RecurrentNet, RecurrentNetWithCNN
>>> model = FCNet(input_size, output_size, num_layers, hidden_size, dropout)
>>> model = RecurrentNet(input_size, hidden_size, num_layers,
        output_size, dropout, bidirectional)
or 
>>> model = RecurrentNetWithCNN(input_size, hidden_size, num_layers,
        output_size, dropout, bidirectional)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from log import setup_custom_logger

logger = setup_custom_logger('Model training')


class FCNet(nn.Module):
    """An fully connected network to predict the emotions induced by the videos."""

    def __init__(self, input_size, output_size, num_layers, hidden_size=-1,
                 dropout=0):
        """Initilialize the RNN.

        Arguments:
            input_size {int} -- size of the input layer
            hidden_size {int} -- size of the hidden layer
            num_layers {int} -- number of layers
            output_size {int} -- ouput size
            dropout {float} -- dropout probability, from 0 to 1
        """
        super(FCNet, self).__init__()

        self._raise_warning_if_necessary(
            input_size, output_size, num_layers, hidden_size, dropout)

        if num_layers == 1:
            self.main = nn.Sequential(
                nn.Linear(input_size, output_size, bias=True),
                nn.ReLU()
            )
        elif num_layers == 2:
            self.main = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size, bias=True),
                nn.ReLU()
            )

        elif num_layers == 3:
            self.main = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size, bias=True),
                nn.ReLU()
            )

        elif num_layers == 4:
            self.main = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size, bias=True),
                nn.ReLU()
            )
        else:
            raise ValueError(
                'num layer should be between 1 and 4. Value {} is given'.format(num_layers))

    def _raise_warning_if_necessary(self, input_size, output_size, num_layers, hidden_size, dropout):
        if num_layers == 1 and dropout != 0:
            logger.warning(
                'With num_layer = 1, the dropout arg (={}) is ignored'.format(dropout))
        if num_layers == 1 and hidden_size != -1:
            logger.warning(
                'With num_layer = 1, the hidden_size arg (={}) is ignored'.format(hidden_size))

    def forward(self, X):
        return self.main(X)


class RecurrentNet(nn.Module):
    """An RNN to predict the emotions induced by the videos."""

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout, bidirectional):
        """Initilialize the RNN.

        Arguments:
            input_size {int} -- size of the input layer
            hidden_size {int} -- size of the hidden layer
            num_layers {int} -- number of layers
            output_size {int} -- ouput size
            dropout {float} -- dropout probability, from 0 to 1
            bidirectional {bool} -- whether the lstm layer is bidirectionnal
        """
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
        """Initilize hidden and cell with random values"""
        hidden = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(
            self.coef*self.num_layers, batch_size, self.hidden_size))

        return hidden, cell


class RecurrentNetWithCNN(nn.Module):
    """An RNN to predict the emotions induced by the videos.

    This RNN has CNN layers for several input features.
    """
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
        "fc6": [1271, 5367],
        "audio": [5367, 6950]
    }

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout, bidirectional):
        """Initilialize the RNN.

        Arguments:
            input_size {int} -- size of the input layer. Deprecated, not used anymore
            hidden_size {int} -- size of the hidden layer
            num_layers {int} -- number of layers for lstm
            output_size {int} -- output size
            dropout {float} -- dropout probability, from 0 to 1
            bidirectional {bool} -- whether the lstm layer is bidirectionnal
        """

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.coef = 2 if self.bidirectional else 1

        super(RecurrentNetWithCNN, self).__init__()

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

        self.cnn_out_dim = 16*13 + 24*3 + 10*2*2 + 32*5 + 28*6 + 4096  # + 1583

        self.lstm_layer = nn.LSTM(input_size=self.cnn_out_dim,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout=dropout,
                                  bias=True,
                                  batch_first=True,
                                  bidirectional=self.bidirectional)

        self.dropout1 = nn.Dropout(dropout)

        self.fc_layer1 = nn.Linear(
            in_features=self.hidden_size*self.coef, out_features=self.hidden_size, bias=True)

        self.fc_layer2 = nn.Linear(
            in_features=self.hidden_size, out_features=output_size, bias=True)

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

        # X_audio = X[:, :, self._features_idx["audio"]
        #            [0]:self._features_idx["audio"][1]]
        # X_audio = torch.reshape(X_audio, (batch_size * seq_len, 1583))

        X = torch.cat((X_acc, X_cedd, X_eh, X_fcth, X_jcd, X_fc6), dim=1)
        X = torch.reshape(X, (batch_size, seq_len, self.cnn_out_dim))

        X, hidden = self.lstm_layer(X, hidden_and_cell)
        X = F.relu(X)
        X = self.dropout1(X)
        X = F.relu(self.fc_layer1(X))
        X = F.relu(self.fc_layer2(X))
        return X

    def initHelper(self, batch_size):
        """Initilize hidden and cell with random values"""
        hidden = Variable(torch.randn(
            self.coef*self.num_layers, batch_size, self.hidden_size))
        cell = Variable(torch.randn(
            self.coef*self.num_layers, batch_size, self.hidden_size))

        return hidden, cell


if __name__ == '__main__':
    from dataset import MediaEval18
    from torch.utils.data import DataLoader

    # Initilize dataset and dataloader
    my_set = MediaEval18(root='./data', seq_len=20,
                         fragment=0.1, features=['all'])
    my_loader = DataLoader(my_set, batch_size=32)

    # Take one batch
    X, Y = next(iter(my_loader))

    # Defines the model
    model_FC = FCNet(
        input_size=X.shape[-1], output_size=2, num_layers=3, dropout=0.8, hidden_size=1024)

    model_without_CNN = RecurrentNet(X.shape[-1], hidden_size=16, num_layers=8,
                                     output_size=2, dropout=0.5, bidirectional=True)

    model_with_CNN = RecurrentNetWithCNN(None, hidden_size=16, num_layers=8,
                                         output_size=2, dropout=0.5, bidirectional=True)

    output = model_FC(X)
    print("output computed with model FC. shape :", output.shape)

    hidden_and_cell = model_without_CNN.initHelper(batch_size=32)
    output = model_without_CNN(X, hidden_and_cell)
    print("output computed with model without CNN. shape :", output.shape)

    hidden_and_cell = model_with_CNN.initHelper(batch_size=32)
    output = model_with_CNN(X, hidden_and_cell)
    print("output computed with model with CNN. shape :", output.shape)
