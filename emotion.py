#!/opt/pytorch/1.0.0/venv/bin/python

import sys
import argparse

parser = argparse.ArgumentParser(description='Train Neural Network for emotion predictions')

parser.add_argument("--seq-len", default=150, type=int, help="Length of a sequence")
parser.add_argument("--num-hidden", default=1, type=int, help="Number of hidden layers in NN")
parser.add_argument("--hidden-dim", default=1000, type=int, help="Dimension of hidden layer")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--input-size", default=6950, type=int, help="Size of the input dataset")
parser.add_argument("--batch-size", default=32, type=int, help="Size of a batch")
parser.add_argument("--grad-clip", default=10, type=float, help="Boundaries of the gradient clipping function (centered on 0)")
parser.add_argument("-S", "--scheduler", default="StepLR", choices=["StepLR", "MultiStepLR", "MultiplicativeLR"], help="Type of scheduler")
parser.add_argument("--nb_batch", default=100, type=int, help="Number of batches")
parser.add_argument("-O", "--optimizer", default="Adam", choices=["Adam", "RMSprop"], help="Type of optimizer")
parser.add_argument("-C", "--crit", default="MSE", choices=["MSE", "Pearson"], help="Typer of criterion for loss computation")
parser.add_argument("-B", "--bidirect", default=False, type=bool, help="Whether to use bidirectional")
parser.add_argument("-R", "--regularisation", choices=["L1", "L2"], help="Type of regularization")
parser.add_argument("-D", "--dropout", default=0, type=float, help="Dropout probability between [0, 1]")

if __name__ == '__main__':
    # TODO: plog argparser to LSTM NN
    args = parser.parse_args()
    # access arguments with getattr(args, 'argument')
    print(args)

    # exit
    sys.exit(0)
