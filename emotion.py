#!/opt/pytorch/1.0.0/venv/bin/python

import sys
import argparse
import torch
from dataset import EmotionDataset
from model import RecurrentNet
from training import MSELoss, trainRecurrentNet
from log import setup_custom_logger

logger = setup_custom_logger("emotion")

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
parser.add_argument("-R", "--regularisation", choices=["L1", "L2"], help="Type of regularization (L1 or L2)")
parser.add_argument("-D", "--dropout", default=0, type=float, help="Dropout probability between [0, 1]")

if __name__ == '__main__':
    try:
        args = parser.parse_args()
        for arg in vars(args):
            logger.info("initialization -- {} - {}".format(arg, getattr(args, arg)))

        device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        logger.info("device - {}".format(str(device)))

        dataset = EmotionDataset()

        model = RecurrentNet(in_dim=getattr(args, 'input_size'),
                            hid_dim=getattr(args, 'hidden_dim'),
                            num_hid=getattr(args, 'num_hidden'),
                            out_dim=2,
                            dropout=getattr(args, 'dropout'))

        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args, 'lr'))

        criterion = MSELoss

        trainRecurrentNet(model=model, dataset=dataset, optimizer=optimizer,
                        criterion=getattr(args, 'crit'),
                        n_batch=100,
                        batch_size=getattr(args, 'batch_size'),
                        seq_len=getattr(args, 'seq_len'),
                        grad_clip=getattr(args, 'grad_clip'),
                        device=device)
    except Exception as exception:
        logger.critical(sys.exc_info())
        
    finally:
        # exit
        sys.exit(0)
