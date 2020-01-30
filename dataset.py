# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:42:18 2020

@author: Tim
"""

import torch
import torch.nn as nn
import read_data
import numpy as np

#%%

class Dataset():
    def __init__(self, batch_size, seq_len,n_films = 66):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_films = n_films
        
        
    def get_random_training_batch(self):
        X = np.zeros((self.batch_size, self.seq_len, self.input_size), dtype = np.float32)
        Y = np.zeros((self.batch_size, self.seq_len, self.input_size), dtype = np.float32)
        
        choice = np.random.randint(0,self.n_films,self.batch_size)
        
        for i,index in enumerate(choice) :
            
            #Choose random starting point to exploit whole sequences
            start = np.random.random()
            X[i,:,:], Y[i,:,:] = read_data.get_window(index, self.seq_len, start)
            #Model predicts one step ahead of the sequence
            
        return X,Y
        
        
    