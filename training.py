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

def MSELoss(batch_predict, batch_label):
    size = list(batch_predict.size())
    batch_predict_reshaped = batch_predict.view(size[0]*size[1], size[2])
    batch_label_reshaped = batch_label.view(size[0]*size[1], size[2])

    loss = torch.nn.MSELoss()
    return loss(batch_predict_reshaped, batch_label_reshaped)

def PearsonLoss(batch_predict, batch_label):
    return 0


def trainRecurrentNet(in_dim, hid_dim, num_hid,out_dim, dropout, n_batch, batch_size, lr, optimizer, seq_len, criterion, grad_clip):
    model = RecurrentNet(in_dim, hid_dim, num_hid,out_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    dataset = EmotionDataset()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    losses = []
    
    if criterion == 'mse':
        loss = MSELoss
    elif criterion == 'pearson':
        loss = PearsonLoss
    
    for batch in range(n_batch):
        #Train mode / optimizer reset
        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        
        X, Y = dataset.get_random_training_batch(batch_size, seq_len)
        X, Y = torch.from_numpy(X, device = device), torch.from_numpy(Y, device = device)
        
        hidden = model.initHelper(batch_size)
        
        y = model(X, hidden)
        
        L = loss(y,Y)
        
        losses.apend(L)
        
        #Backward step
        L.backward()
            
        #Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            
        #Optimizer step
        optimizer.step()
        
        if batch%10 == 0:
            print(f'Batch : {batch}')
            print(f"\n Loss : {L : 3f}")
            
        if batch%20 == 0:
            torch.save(model, 'models/RecurrentNet.pt')
    #TODO Add Loss plotting
    torch.save(model, 'models/RecurrentNet.pt')

        
        
        