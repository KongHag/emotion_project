# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:32:25 2020

@author: lucas
"""
import torch
from scipy.stats import pearsonr

def get_metrics(model, testloader):
    
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model.eval()
    
    inputs = torch.tensor([])
    outputs = torch.tensor([])
    labels = torch.tensor([])
    
    for idx_batch, (X, Y) in enumerate(testloader):

        # Copy to GPU
        gpu_X = X.to(device=device, dtype=torch.float32)
        
        
        # Init hidden layer input, if necessary
        try:
            # If there is a function initHelper
            hidden, cell = model.initHelper(gpu_X.shape[0])
            gpu_hidden = hidden.to(device=device)
            gpu_cell = cell.to(device=device)
            model_args = (gpu_X, (gpu_hidden, gpu_cell))
        except:
            # The model does not need a hidden layer init
            model_args = (gpu_X,)
    
        # Output and loss computation
        gpu_output = model(*model_args)
        
        flattenX = torch.flatten(X, end_dim=1).detach()
        flattenY = torch.flatten(Y, end_dim=1).detach()
        flattenOutput = torch.flatten(gpu_output.to(device=torch.device('cpu')), end_dim=1).detach()
        inputs = torch.cat((inputs, flattenX), 0)
        outputs = torch.cat((outputs, flattenOutput), 0)
        labels = torch.cat((labels, flattenY), 0)
    
    predictions_valence, predictions_arousal = outputs[0][:], outputs[1][:]
    label_valence, label_arousal = labels[0][:], labels[1][:]
    
    MSE = torch.nn.MSELoss()
    MSE_valence = str(float(MSE(predictions_valence, label_valence)))
    MSE_arousal = str(float(MSE(predictions_arousal, label_arousal)))
    
    r_valence = pearsonr(predictions_valence, label_valence)[0]
    r_arousal = pearsonr(predictions_arousal, label_arousal)[0]
    
    results = {
        "MSE_valence" : MSE_valence,
        "MSE_arousal" : MSE_arousal,
        "r_valence" : r_valence,
        "r_arousal" : r_arousal
    }
    return results


    
    
    
    