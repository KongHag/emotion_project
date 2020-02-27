# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:01:37 2020

@author: Tim
"""

from dataset import EmotionDataset
from dataset import MediaEval18
from model import RecurrentNet
import numpy as np
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

def MSELoss_V_A(batch_predict, batch_label):
    size = list(batch_predict.size())
    batch_predict_reshaped_V = batch_predict.view(-1, size[2])[:,0]
    batch_label_reshaped_V = batch_label.view(-1, size[2])[:,0]
    batch_predict_reshaped_A = batch_predict.view(-1, size[2])[:,1]
    batch_label_reshaped_A = batch_label.view(-1, size[2])[:,1]
    loss = torch.nn.MSELoss()
    
    return loss(batch_predict_reshaped_V, batch_label_reshaped_V), loss(batch_predict_reshaped_A, batch_label_reshaped_A)


def PearsonCoefficient(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def Pearson_V_A(batch_predict, batch_label):
    size = list(batch_predict.size())
    batch_predict_reshaped_V = batch_predict.view(-1, size[2])[:,0]
    batch_label_reshaped_V = batch_label.view(-1, size[2])[:,0]
    batch_predict_reshaped_A = batch_predict.view(-1, size[2])[:,1]
    batch_label_reshaped_A = batch_label.view(-1, size[2])[:,1]

    pearson_V = PearsonCoefficient(batch_predict_reshaped_V, batch_label_reshaped_V)
    pearson_A = PearsonCoefficient(batch_predict_reshaped_A, batch_label_reshaped_A)
    return pearson_V, pearson_A



def store(model):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f='./models/RecurrentNet.pt')


def compute_test_loss(model, testloader, optimizer, criterion, device):
    losses = []
    eval_losses = []
    for idx_batch, (X, Y) in enumerate(testloader):
        logger.debug("Starting testing with batch {}".format(idx_batch))

        model.train()
        optimizer.zero_grad()
        model.zero_grad()

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
        V,A = MSELoss_V_A(gpu_output, gpu_Y)
        r_V, r_A = PearsonCoefficient(gpu_output, gpu_Y) 
        eval_losses.append([V,A,r_V,r_A])
        eval_losses = torch.tensor(eval_losses, device=device).float()
        losses.append(float(loss))
        logger.debug("loss computed : {}".format(loss))
    means = torch.mean(eval_losses,dim = 0)
    return np.mean(losses), means


def trainRecurrentNet(model, trainloader, testloader, optimizer, criterion,
                      nb_epoch, grad_clip, device):
    logger.info("start training")
    if criterion == "MSE":
        criterion = MSELoss
    elif criterion == "Pearson":
        criterion = PearsonCoefficient


    train_losses, test_losses = [], []
    for epoch in range(nb_epoch):
        logger.info("Starting training epoch {}".format(epoch))
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

            # Init hidden layer input
            hidden, cell = model.initHelper(gpu_X.shape[0])
            gpu_hidden = hidden.to(device=device)
            gpu_cell = cell.to(device=device)
            logger.debug("hidden layer and cell initialized")

            # Output and loss computation
            gpu_output = model(gpu_X, (gpu_hidden, gpu_cell))
            logger.debug("output computed")
            loss = criterion(gpu_output, gpu_Y)
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

        logger.info(f'Epoch : {epoch}')
        train_losses.append((idx_batch, float(loss)))
        test_loss, eval_loss = compute_test_loss(
                model, testloader, optimizer, criterion, device)
        test_losses.append((idx_batch, test_loss))
        logger.info(f"Test loss : {test_loss : 3f}")
        logger.info(f"Train loss : {loss : 3f}")
        logger.info("Eval loss : MSE Valence : {0}, MSE Arousal : {1}, Pearson Valence {2}, Pearson Arousal {3}".format(*eval_loss))
        

        if epoch % 20 == 0:
            pickle.dump(train_losses, open("data/train_losses.pickle", "wb"))
            pickle.dump(test_losses, open("data/test_losses.pickle", "wb"))
            store(model)

    # TODO Add Loss plotting
    torch.save(model.state_dict(), f='./models/RecurrentNet.pt')


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    trainset = MediaEval18(root='./data', train=True, seq_len=100, nb_sequences=100, shuffle=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True)
    testset = MediaEval18(root='./data', train=False, seq_len=100,
                          nb_sequences=8, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True)

    model = RecurrentNet(in_dim=6950, hid_dim=100, num_hid=2, out_dim=2,
                         dropout=0.5)
    logger.info("neural network : {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = MSELoss
 
    trainRecurrentNet(model=model, trainloader=trainloader, testloader=testloader,
                      optimizer=optimizer, criterion=criterion, nb_epoch=30,
                      grad_clip=10, device=device)

