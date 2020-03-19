# -*- coding: utf-8 -*-
"""
TODO : write a docstring
"""

import sys
import os
import time
import argparse
import torch
from dataset import MediaEval18
from torch.utils.data import DataLoader
from model import RecurrentNet
from training import train_model
from log import setup_custom_logger
import logging
import json

logger = setup_custom_logger("emotion")

parser = argparse.ArgumentParser(
    description='Train Neural Network for emotion predictions')

parser.add_argument("--seq-len", default=20, type=int,
                    help="Length of a sequence")
parser.add_argument("--num-hidden", default=1, type=int,
                    help="Number of hidden layers in NN")
parser.add_argument("--hidden-size", default=32, type=int,
                    help="Dimension of hidden layer")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=128,
                    type=int, help="Size of a batch")
parser.add_argument("--grad-clip", default=None, type=float,
                    help="Gradient clipped between [- grad-clip, grad-clip]")
# TODO : implement scheduler
# parser.add_argument("-S", "--scheduler", default="StepLR", choices=["StepLR", "MultiStepLR", "MultiplicativeLR"], help="Type of scheduler")
parser.add_argument("--nb-epoch", default=100,
                    type=int, help="Number of epoch")
parser.add_argument("-O", "--optimizer", default="SGD",
                    choices=["Adam", "RMSprop", "SGD"], help="Type of optimizer")
parser.add_argument("-C", "--crit", default="MSE",
                    choices=["MSE", "Pearson"], help="Type of criterion for loss computation")
# TODO : implement bidirect
# parser.add_argument("-B", "--bidirect", default=False,
#                     type=bool, help="Whether to use bidirectional")
parser.add_argument("--weight-decay", default=0, type=float,
                    help="L2 regularization coefficient")
parser.add_argument("-D", "--dropout", default=0, type=float,
                    help="Dropout probability between [0, 1]")
parser.add_argument("--logger-level", default=20, type=int,
                    help="Logger level: from 10 (debug) to 50 (critical)")
parser.add_argument("--fragment", default=1, type=float,
                    help="The percentage of the dataset used. From 0 to 1")


def run(config):
    # Select device
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("device - {}".format(str(device)))

    # Dataset initilisation
    trainset = MediaEval18(
        root='./data', train=True, seq_len=config['seq_len'],
        shuffle=True, fragment=config['fragment'], features=['all'])
    trainloader = DataLoader(trainset, batch_size=config['batch_size'],
        shuffle=True, num_workers=8)
    logger.info(
        "trainset/loader initialized : trainset lenght : {}".format(len(trainset)))

    testset = MediaEval18(
        root='./data', train=False, seq_len=config['seq_len'],
        shuffle=True, fragment=config['fragment'], features=['all'])
    testloader = DataLoader(testset, batch_size=config['batch_size'],
                            num_workers=8)
    logger.info(
        "testset/loader initialized : testset lenght : {}".format(len(testset)))

    # Model initilisation
    model = RecurrentNet(input_size=next(iter(trainset))[0].shape[1],
                         hidden_size=config['hidden_size'],
                         num_layers=config['num_hidden'],
                         output_size=2,
                         dropout=config['dropout'],
                         bidirectional=False)
    model.to(device)
    logger.info("model : {}".format(model))

    # Define criterion
    criterion = torch.nn.MSELoss()
    logger.info("criterion : {}".format(criterion))

    # Define optimizer
    attr_optimizer = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    if attr_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    if attr_optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    if attr_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info("optimizer : {}".format(optimizer))

    # Train model
    train_losses, test_losses = train_model(
        model=model, trainloader=trainloader, testloader=testloader,
        criterion=criterion, optimizer=optimizer, device=device,
        grad_clip=config['grad_clip'],
        nb_epoch=config['nb_epoch'])

    return train_losses, test_losses


def save_config_and_results(config, train_losses, test_losses):
    """Save in a file in results/ the config and the results"""
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }

    config_and_results = {**config, **results}

    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = "results/emotion_" + \
        time.strftime("%Y-%m-%d_%H:%M:%S") + ".json"

    # Save the config and the results
    with open(file_name, 'w') as file:
        json.dump(config_and_results, file)

if __name__=='__main__':
    # Parse args
    config = vars(parser.parse_args())
    logger = logging.getLogger()
    logger.setLevel(config['logger_level'])

    for arg_name, arg in config.items():
        logger.info(
            "initialization -- {} - {}".format(arg_name, arg))

    try:
        train_losses, test_losses = run(config)
        save_config_and_results(config, train_losses, test_losses)

    except Exception as exception:
        logger.critical(sys.exc_info())
    finally:
        # exit
        sys.exit(0)
