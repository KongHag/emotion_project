# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:37 2020

@author: Tim
"""

from dataset import EmotionDataset
from model import RecurrentNet
import torch
from log import setup_custom_logger
# import numpy as np
# from tqdm import tqdm
# import torch.nn as nn
# import sys
# sys.path.insert(0, '.')
# sys.path.insert(0, '..')
logger = setup_custom_logger('Model training')
# %%


def MSELoss(batch_predict, batch_label):
    size = list(batch_predict.size())
    batch_predict_reshaped = batch_predict.view(-1, size[2])
    batch_label_reshaped = batch_label.view(-1, size[2])

    loss = torch.nn.MSELoss()
    return loss(batch_predict_reshaped, batch_label_reshaped)


def PearsonLoss(batch_predict, batch_label):
    return 0


def trainRecurrentNet(model, dataset, optimizer, criterion, n_batch, batch_size,
                      seq_len, grad_clip, device):
    losses = []
    for idx_batch in range(n_batch):
        # Train mode / optimizer reset
        model.train()
        optimizer.zero_grad()
        model.zero_grad()

        # Load numpy arrays
        X, Y = dataset.get_random_training_batch(batch_size, seq_len)

        # Copy to GPU
        X, Y = torch.from_numpy(X).to(
            device=device), torch.from_numpy(Y).to(device=device)

        # Init hidden layer input
        hidden, cell = model.initHelper(batch_size)
        hidden.to(device=device)
        cell.to(device=device)
        hidden = (hidden, cell)

        # Output and loss computation
        y = model(X, hidden)
        loss = criterion(y, Y)
        losses.append(loss)

        # Backward step
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()

        if idx_batch % 10 == 0:
            logger.info(f'Batch : {idx_batch}')
            logger.info(f" Loss : {loss : 3f}")

        if idx_batch % 20 == 0:
            torch.save(model.state_dict(), f='./models/RecurrentNet.pt')

    # TODO Add Loss plotting
    torch.save(model.state_dict(), f='./models/RecurrentNet.pt')

if __name__=='__main__':
#    device = torch.device(
#        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#
#    dataset = EmotionDataset()
#
#    model = RecurrentNet(in_dim=6950, hid_dim=100, num_hid=2, out_dim=2,
#                        dropout=0.5)
#
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#    criterion = MSELoss
#
#    trainRecurrentNet(model=model, dataset=dataset, optimizer=optimizer,
#                    criterion=criterion, n_batch=100, batch_size=30, seq_len=100,
#                    grad_clip=10, device=device)
    logger.info("test")
