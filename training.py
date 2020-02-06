# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:37 2020

@author: Tim
"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import sys 
sys.path.insert(0,'.')
sys.path.insert(0,'..')
from model import RecurrentNet
from dataset import EmotionDataset
#%%


def trainRecurrentNet(in_dim, hid_dim, num_hid,out_dim, dropout, n_batch, batch_size, lr, optimizer, seq_len):
    model = RecurrentNet(in_dim, hid_dim, num_hid,out_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    dataset = EmotionDataset(seq_len)
    
    for b in range(n_batch):
        
        
        #Train mode / optimizer reset
        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        
        