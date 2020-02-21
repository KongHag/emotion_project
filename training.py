# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:37 2020

@author: Tim
"""

from dataset import EmotionDataset
from model import RecurrentNet
import torch
from log import setup_custom_logger
import pickle
import os

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


def store(model):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f='./models/RecurrentNet.pt')


def trainRecurrentNet(model, dataset, optimizer, criterion, n_batch, batch_size,
                      seq_len, grad_clip, device):

    logger.info("start training")
    if criterion == "MSE":
        criterion = MSELoss
    elif criterion == "Pearson":
        criterion = PearsonLoss

    losses = []
    for idx_batch in range(n_batch):
        logger.debug("Starting training with batch {}".format(idx_batch))

        # Train mode / optimizer reset
        model.train()
        logger.debug("model trained")
        optimizer.zero_grad()
        model.zero_grad()
        logger.debug("gradients cleared")

        # Load numpy arrays
        X, Y = dataset.get_random_training_batch(batch_size, seq_len)
        logger.debug("batch generated")

        # Copy to GPU
        gpu_X = torch.from_numpy(X).to(device=device)
        gpu_Y = torch.from_numpy(Y).to(device=device)
        logger.debug("X, Y copied on device {}".format(device))

        # Init hidden layer input
        hidden, cell = model.initHelper(batch_size)
        gpu_hidden = hidden.to(device=device)
        gpu_cell = cell.to(device=device)
        logger.debug("hidden layer and cell initialized")

        # Output and loss computation
        gpu_output = model(gpu_X, (gpu_hidden, gpu_cell))
        logger.debug("output computed")
        loss = criterion(gpu_output, gpu_Y)
        losses.append(loss)
        logger.debug("loss computed : {}".format(loss))

        # Backward step
        loss.backward()
        logger.debug("loss backwarded")

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        logger.debug("grad clipped")

        # Optimizer step
        optimizer.step()
        logger.debug("optimizer steped")

        pickle.dump(losses, open("data/losses.pickle", "wb"))

        if idx_batch % 10 == 0:
            logger.info(f'Batch : {idx_batch}')
            logger.info(f"Loss : {loss : 3f}")

        if idx_batch % 20 == 0:
            store(model)

    # TODO Add Loss plotting
    torch.save(model.state_dict(), f='./models/RecurrentNet.pt')


if __name__ == '__main__':
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset = EmotionDataset()

    model = RecurrentNet(in_dim=6950, hid_dim=100, num_hid=2, out_dim=2,
                         dropout=0.5)
    logger.info("neural network : {}".format(net))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = MSELoss

    trainRecurrentNet(model=model, dataset=dataset, optimizer=optimizer,
                      criterion=criterion, n_batch=100, batch_size=30, seq_len=100,
                      grad_clip=10, device=device)
