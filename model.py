# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:46:42 2020

@author: Tim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(6950, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
