#!/opt/pytorch/1.0.0/venv/bin/python

import sys
import argparse
import torch
from dataset import MediaEval18
from model import RecurrentNet
from training import MSELoss, trainRecurrentNet
from log import setup_custom_logger
import logging

logger = setup_custom_logger("emotion")

parser = argparse.ArgumentParser(
    description='Train Neural Network for emotion predictions')

parser.add_argument("--seq-len", default=100, type=int,
                    help="Length of a sequence")
parser.add_argument("--num-hidden", default=1, type=int,
                    help="Number of hidden layers in NN")
parser.add_argument("--hidden-dim", default=1000, type=int,
                    help="Dimension of hidden layer")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--input-size", default=6950, type=int,
                    help="Size of the input dataset")
parser.add_argument("--batch-size", default=32,
                    type=int, help="Size of a batch")
parser.add_argument("--grad-clip", default=10, type=float,
                    help="Boundaries of the gradient clipping function (centered on 0)")
# parser.add_argument("-S", "--scheduler", default="StepLR", choices=["StepLR", "MultiStepLR", "MultiplicativeLR"], help="Type of scheduler")
parser.add_argument("--nb-epoch", default=100,
                    type=int, help="Number of epoch")
parser.add_argument("-O", "--optimizer", default="Adam",
                    choices=["Adam", "RMSprop"], help="Type of optimizer")
parser.add_argument("-C", "--crit", default="MSE",
                    choices=["MSE", "Pearson"], help="Typer of criterion for loss computation")
parser.add_argument("-B", "--bidirect", default=False,
                    type=bool, help="Whether to use bidirectional")
parser.add_argument("-R", "--regularisation",
                    choices=["L1", "L2"], help="Type of regularization (L1 or L2)")
parser.add_argument("-D", "--dropout", default=0, type=float,
                    help="Dropout probability between [0, 1]")
parser.add_argument("--logger-level", default=20, type=int,
                    help="logger level from 10 (debug) to 50 (critical)")


def run(args):
    # Select device
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("device - {}".format(str(device)))

    # Init dataset
    trainset = MediaEval18(
        root='./data', train=True, seq_len=getattr(args, 'seq_len'))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=getattr(args, 'batch_size'), shuffle=True,
        num_workers=10)
    testset = MediaEval18( 
        root='./data', train=False, seq_len=getattr(args, 'seq_len'), nb_sequences=256 ,shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=getattr(args, 'batch_size'), shuffle=True,
        num_workers=10)
    logger.debug("dataset/dataloader initialized")

    # Model initilisation
    model = RecurrentNet(in_dim=getattr(args, 'input_size'),
                         hid_dim=getattr(args, 'hidden_dim'),
                         num_hid=getattr(args, 'num_hidden'),
                         out_dim=2,
                         dropout=getattr(args, 'dropout'))
    model.to(device)
    logger.info("model : {}".format(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=getattr(args, 'lr'))
    logger.debug("optimizer : {}".format(optimizer))

    criterion = MSELoss
    logger.debug("criterion : {}".format(criterion))

    trainRecurrentNet(model=model,
                      trainloader=trainloader, testloader=testloader,
                      optimizer=optimizer,
                      criterion=getattr(args, 'crit'),
                      nb_epoch=getattr(args, 'nb_epoch'),
                      grad_clip=getattr(args, 'grad_clip'),
                      device=device)


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    for arg in vars(args):
        logger.info(
            "initialization -- {} - {}".format(arg, getattr(args, arg)))

    logger = logging.getLogger()
    logger.setLevel(getattr(args, 'logger_level'))

    try:
        run(args)
    except Exception as exception:
        logger.critical(sys.exc_info())
    finally:
        # exit
        sys.exit(0)
