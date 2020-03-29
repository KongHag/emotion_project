# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:32:25 2020

@author: lucas
"""
import torch
from scipy.stats import pearsonr

def metrics(model, inputs, labels):
    model.eval()
    predictions_valence, predictions_arousal = model(inputs)
    label_valence, label_arousal = labels
    
    MSE = torch.nn.MSELoss()
    MSE_valence = MSE(predictions_valence, label_valence)
    MSE_arousal = MSE(predictions_arousal, label_arousal)
    
    r_valence = pearsonr(predictions_valence, label_valence)
    r_arousal = pearsonr(predictions_arousal, label_arousal)
    
    return MSE_arousal, MSE_valence, r_arousal, r_valence
    
    
    
    