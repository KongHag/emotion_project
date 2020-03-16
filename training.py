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


def do_one_epoch(model, loader, train, criterion, device,
                 optimizer=None, grad_clip=None):

    losses = []
    for idx_batch, (X, Y) in enumerate(loader):
        logger.debug("Starting with batch {}".format(idx_batch))

        if train:
            model.train()
            logger.debug("model in train mode")
        else:
            model.eval()
            logger.debug("model in eval mode")

        # Zero optimizer grad
        if train and optimizer is not None:
            optimizer.zero_grad()
            logger.debug("optimizer gradient cleared")

        # Zero model grad
        if train:
            model.zero_grad()
            logger.debug("model gradient cleared")

        # Copy to GPU
        gpu_X = X.to(device=device, dtype=torch.float32)
        gpu_Y = Y.to(device=device, dtype=torch.float32)
        logger.debug("X, Y copied on device {}".format(device))

        # Output and loss computation
        gpu_output = model(gpu_X)
        logger.debug("output computed")
        loss = criterion(gpu_output, gpu_Y)
        logger.debug("loss computed : {}".format(float(loss)))
        losses.append(float(loss))

        if train:
            # Backward step
            loss.backward()
            logger.debug("loss backwarded")

            # Gradient clip
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                logger.debug("grad clipped")

            if optimizer is not None:
                # Optimizer step
                optimizer.step()
                logger.debug("optimizer steped")

    return np.mean(losses)


def train_model(model, trainloader, testloader, criterion, optimizer, device,
                grad_clip, nb_epoch):
    """Run this function to train the model"""

    logger.info("start training network")

    for epoch in range(nb_epoch):
        logger.info("Starting training epoch {}".format(epoch))

        # Test the model
        test_loss = do_one_epoch(model=model, loader=testloader, train=False,
                                 criterion=criterion, device=device,
                                 grad_clip=grad_clip)

        # Train the model
        train_loss = do_one_epoch(model=model, loader=trainloader, train=True,
                                  criterion=criterion, device=device,
                                  optimizer=optimizer, grad_clip=grad_clip)

        logger.info(f"Test  loss : {test_loss : 3f}")
        logger.info(f"Train loss : {train_loss : 3f}")


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Define sets then loaders
    trainset = MediaEval18(root='./data', train=True, fragment=0.3, shuffle=True,
                           features=["all"])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True)

    testset = MediaEval18(root='./data', train=False, fragment=0.3, shuffle=True,
                          features=["all"])
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True)

    # Define the model
    model = FCNet()
    logger.info("neural network : {}".format(model))

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Define criterion
    criterion = torch.nn.MSELoss()

    # Train the network
    train_model(model=model, trainloader=trainloader, testloader=testloader,
                criterion=criterion, optimizer=optimizer, device=device,
                grad_clip=10, nb_epoch=30)
