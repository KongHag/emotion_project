# -*- coding: utf-8 -*-
"""
Defines executable to run a training of a Recurrent net based on given parameters

This will
    - create a RecurrentNet based on input parameters
    - train it with the input data of Liris dataset,
    - log the train loss and test loss at each epoch in the log file
    - plot the train_loss x epoch and test_loss x epoch graph in /results folder

How to use :
> python emotion.py [-h] [--add-CNN] [--seq-len SEQ_LEN]
             [--num-hidden NUM_HIDDEN] [--hidden-size HIDDEN_SIZE]
             [--lr LR] [--batch-size BATCH_SIZE] [--grad-clip GRAD_CLIP]
             [--nb-epoch NB_EPOCH] [-O {Adam,RMSprop,SGD}] [-B BIDIRECT]
             [--weight-decay WEIGHT_DECAY] [-D DROPOUT]
             [--logger-level LOGGER_LEVEL] [--fragment FRAGMENT]
             [--features {acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} [{acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} ...]]
             [--no-overlapping]
"""

import sys
import os
import time
import argparse
import torch
from dataset import MediaEval18
from torch.utils.data import DataLoader
from model import RecurrentNet, RecurrentNetWithCNN, FCNet
from training import train_model
from log import setup_custom_logger
from metrics import get_metrics
import logging
import json

logger = setup_custom_logger("emotion")
features = MediaEval18._features_len.keys()

parser = argparse.ArgumentParser(
    description='Train Neural Network for emotion predictions')

parser.add_argument("--model", default="LSTM",
                    choices=["FC", "LSTM", "CNN_LSTM"], help="Type of model")
parser.add_argument("--seq-len", default=20, type=int,
                    help="Length of a sequence")
parser.add_argument("--num-hidden", default=2, type=int,
                    help="Number of hidden layers in NN")
parser.add_argument("--hidden-size", default=32, type=int,
                    help="Dimension of hidden layer")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=8,
                    type=int, help="Size of a batch")
parser.add_argument("--grad-clip", default=None, type=float,
                    help="Gradient clipped between [- grad-clip, grad-clip]")
# TODO : implement scheduler
# parser.add_argument("-S", "--scheduler", default="StepLR", choices=["StepLR", "MultiStepLR", "MultiplicativeLR"], help="Type of scheduler")
parser.add_argument("--nb-epoch", default=100,
                    type=int, help="Number of epoch")
parser.add_argument("-O", "--optimizer", default="SGD",
                    choices=["Adam", "RMSprop", "SGD"], help="Type of optimizer")
# Pearson not implemented
# parser.add_argument("-C", "--crit", default="MSE",
#                     choices=["MSE", "Pearson"], help="Type of criterion for loss computation")
parser.add_argument("-B", "--bidirect", default=False,
                    type=bool, help="Whether to use bidirectional")
parser.add_argument("--weight-decay", default=0, type=float,
                    help="L2 regularization coefficient")
parser.add_argument("-D", "--dropout", default=0, type=float,
                    help="Dropout probability between [0, 1]")
parser.add_argument("--logger-level", default=20, type=int,
                    help="Logger level: from 10 (debug) to 50 (critical)")
parser.add_argument("--fragment", default=1, type=float,
                    help="The percentage of the dataset used. From 0 to 1")
parser.add_argument("--features", default="all", nargs='+',
                    choices=features, help="Features used")
parser.add_argument('--no-overlapping', dest='overlapping', action='store_false',
                    help='Forbid overlapping between sequences in dataset')
parser.set_defaults(overlapping=True)


def run(config):
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(config['logger_level'])

    # Log config
    for arg_name, arg in config.items():
        logger.info(
            "initialization -- {} - {}".format(arg_name, arg))

    # Select device
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("device - {}".format(str(device)))

    # Dataset initilisation
    trainset = MediaEval18(
        root='./data', train=True, seq_len=config['seq_len'],
        shuffle=True, fragment=config['fragment'], features=config['features'], overlapping=config['overlapping'])
    trainloader = DataLoader(trainset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=8)
    logger.info(
        "trainset/loader initialized : trainset lenght : {}".format(len(trainset)))

    testset = MediaEval18(
        root='./data', train=False, seq_len=config['seq_len'],
        shuffle=True, fragment=config['fragment'], features=config['features'], overlapping=config['overlapping'])
    testloader = DataLoader(testset, batch_size=config['batch_size'],
                            num_workers=8)
    logger.info(
        "testset/loader initialized : testset lenght : {}".format(len(testset)))

    # Model initilisation
    if config['model'] == 'FC':
        model = FCNet(
            input_size=next(iter(trainset))[0].shape[1],
            output_size=2,
            num_hidden=config['num_hidden'],
            hidden_size=config.get('hidden_size', -1),
            dropout=config.get('dropout', 0))
    elif config['model'] == 'LSTM':
        model = RecurrentNet(
            input_size=next(iter(trainset))[0].shape[1],
            hidden_size=config.get('hidden_size', -1),
            num_layers=config['num_hidden'],
            output_size=2,
            dropout=config.get('dropout', 0),
            bidirectional=config['bidirect'])
    elif config['model'] == 'CNN_LSTM':
        model = RecurrentNetWithCNN(
            input_size=next(iter(trainset))[0].shape[1],
            hidden_size=config.get('hidden_size', -1),
            num_layers=config['num_hidden'],
            output_size=2,
            dropout=config.get('dropout', 0),
            bidirectional=config['bidirect'])
    model.to(device)
    logger.info("model : {}".format(model))
    logger.info('number of param : {}'.format(
        sum(p.numel() for p in model.parameters())))
    logger.info('number of learnable param : {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

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
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    logger.info("optimizer : {}".format(optimizer))

    # Train model
    train_losses, test_losses = train_model(
        model=model, trainloader=trainloader, testloader=testloader,
        criterion=criterion, optimizer=optimizer, device=device,
        grad_clip=config['grad_clip'],
        nb_epoch=config['nb_epoch'])
    logger.info("training done")

    metrics = get_metrics(model, testloader)

    save_config_and_results(config, train_losses, test_losses, metrics)


def save_config_and_results(config, train_losses, test_losses, metrics):
    """Save in a file in results/ the config and the results"""
    results = {
        'MSE_valence': metrics["MSE_valence"],
        'MSE_arousal': metrics["MSE_arousal"],
        'r_valence': metrics["r_valence"],
        'r_arousal': metrics["r_arousal"],
        'train_losses': train_losses,
        'test_losses': test_losses
    }

    config_and_results = {**config, **results}

    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = "results/emotion_" + \
        time.strftime("%Y-%m-%d_%H-%M-%S") + ".json"

    # Save the config and the results
    with open(file_name, 'w') as file:
        json.dump(config_and_results, file)
    logger.info("results saved")


if __name__ == '__main__':
    # Parse args
    config = vars(parser.parse_args())

    try:
        run(config)

    except Exception as exception:
        logger.critical(sys.exc_info())
    finally:
        # exit
        sys.exit(0)
