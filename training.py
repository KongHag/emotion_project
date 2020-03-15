# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:37 2020

@author: Tim
"""

from dataset import MediaEval18
from model import FCNet
import numpy as np
import torch
from log import setup_custom_logger
import pickle
import os

logger = setup_custom_logger('Model training')


def test(model, testloader, criterion, grad_clip, device):

    test_losses = []
    for idx_batch, (X, Y) in enumerate(testloader):
        logger.debug("Starting training with batch {}".format(idx_batch))

        # Train mode / optimizer reset
        model.eval()
        logger.debug("model in train mode")

        # Copy to GPU
        gpu_X = X.to(device=device, dtype=torch.float32)
        gpu_Y = Y.to(device=device, dtype=torch.float32)
        logger.debug("X, Y copied on device {}".format(device))

        # Output and loss computation
        gpu_output = model(gpu_X)
        logger.debug("output computed")
        loss = criterion(gpu_output, gpu_Y)
        logger.debug("loss computed : {}".format(float(loss)))
        test_losses.append(float(loss))

    return np.mean(test_losses)

def train(model, trainloader, optimizer, criterion, grad_clip, device):
    train_losses = []
    for idx_batch, (X, Y) in enumerate(trainloader):
        logger.debug("Starting training with batch {}".format(idx_batch))

        # Train mode / optimizer reset
        model.train()
        logger.debug("model in train mode")
        optimizer.zero_grad()
        model.zero_grad()
        logger.debug("gradients cleared")

        # Copy to GPU
        gpu_X = X.to(device=device, dtype=torch.float32)
        gpu_Y = Y.to(device=device, dtype=torch.float32)
        logger.debug("X, Y copied on device {}".format(device))

        # Output and loss computation
        gpu_output = model(gpu_X)
        logger.debug("output computed")
        loss = criterion(gpu_output, gpu_Y)
        logger.debug("loss computed : {}".format(float(loss)))
        train_losses.append(float(loss))

        # Backward step
        loss.backward()
        logger.debug("loss backwarded")

        # Gradient clip
        if grad_clip != None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            logger.debug("grad clipped")

        # Optimizer step
        optimizer.step()
        logger.debug("optimizer steped")

    return np.mean(train_losses)
  

def train_network(model, trainloader, testloader, optimizer, criterion,
                      nb_epoch, grad_clip, device):
    logger.info("start training")

    for epoch in range(nb_epoch):
        logger.info("Starting training epoch {}".format(epoch))
        test_loss = test(model, testloader, criterion, grad_clip, device)
        train_loss = train(model, trainloader, optimizer, criterion, grad_clip, device)

        logger.info(f"Test  loss : {test_loss : 3f}")
        logger.info(f"Train loss : {train_loss : 3f}")



if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    trainset = MediaEval18(root='./data', train=True, fragment = 0.3, shuffle=True,
                           features=["all"])
    print(len(trainset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True)

    testset = MediaEval18(root='./data', train=False, fragment = 0.3, shuffle=True,
                          features=["all"])
    print(len(testset))

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True)
    
    # #elmt en commun 
    # com = [elmt for elmt in trainset._possible_imgs if elmt in testset._possible_imgs]
    # print(com)
    # for elmt in com:
    #     trainset._possible_imgs.remove(elmt)
    #     testset._possible_imgs.remove(elmt)

    model = FCNet()
    logger.info("neural network : {}".format(model))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    criterion = torch.nn.MSELoss()

    train_network(model=model, trainloader=trainloader, testloader=testloader,
                      optimizer=optimizer, criterion=criterion, nb_epoch=30,
                      grad_clip=10, device=device)
