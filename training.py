# -*- coding: utf-8 -*-
"""
Defines the training & testing phase of a model

Parameters for the training phase (optimizer, criterion, ...)
are setup as argumetns when executing emotion.py

For each epoch & for each batch in the dataset :
    - send tensors to gpu if available
    - compute output of network
    - backpropagate if training
> return the test loss and training loss

How to use :
>>> train_losses, test_losses = train_model(
            model=model, trainloader=trainloader, testloader=testloader,
            criterion=criterion, optimizer=optimizer, device=device,
            grad_clip=config['grad_clip'],
            nb_epoch=config['nb_epoch']
        )
"""

import numpy as np
import torch
from log import setup_custom_logger

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

        # Init hidden layer input
        hidden, cell = model.initHelper(gpu_X.shape[0])
        gpu_hidden = hidden.to(device=device)
        gpu_cell = cell.to(device=device)
        logger.debug("hidden layer and cell initialized")

        # Output and loss computation
        gpu_output = model(gpu_X, (gpu_hidden, gpu_cell))
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
    train_losses, test_losses = [], []

    for epoch in range(nb_epoch):
        logger.info("Starting training epoch {}".format(epoch))

        # Test the model
        test_loss = do_one_epoch(model=model, loader=testloader, train=False,
                                 criterion=criterion, device=device,
                                 grad_clip=grad_clip)
        logger.info(f"Test  loss : {test_loss : 3f}")
        test_losses.append(test_loss)

        # Train the model
        train_loss = do_one_epoch(model=model, loader=trainloader, train=True,
                                  criterion=criterion, device=device,
                                  optimizer=optimizer, grad_clip=grad_clip)
        logger.info(f"Train loss : {train_loss : 3f}")
        train_losses.append(train_loss)

    return train_losses, test_losses


if __name__ == '__main__':
    import logging
    from dataset import MediaEval18
    from torch.utils.data import DataLoader
    from model import RecurrentNet

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Define sets then loaders
    trainset = MediaEval18(root='./data', train=True, fragment=0.3,
                           shuffle=True, features=["all"])
    trainloader = DataLoader(
        trainset, batch_size=4, shuffle=True)

    testset = MediaEval18(root='./data', train=False, fragment=0.3,
                          shuffle=True, features=["all"])
    testloader = DataLoader(
        testset, batch_size=4, shuffle=True)

    # Define the model
    model = RecurrentNet(input_size=next(iter(trainset))[0].shape[1],
                         hidden_size=32,
                         num_layers=1,
                         output_size=2,
                         dropout=0,
                         bidirectional=False)

    logger.info("neural network : {}".format(model))

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Define criterion
    criterion = torch.nn.MSELoss()

    # Train the network
    train_model(model=model, trainloader=trainloader, testloader=testloader,
                criterion=criterion, optimizer=optimizer, device=device,
                grad_clip=10, nb_epoch=30)
