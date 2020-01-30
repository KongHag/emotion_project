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
    def __init__(self, batch_size, seq_len,n_films = 66, input_size = 1583+5367):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_films = n_films
        self.input_size = input_size
        self.VA = 2
        
        
    def get_random_training_batch(self):
        X = np.zeros((self.batch_size, self.seq_len, self.input_size), dtype = np.float32)
        Y = np.zeros((self.batch_size, self.seq_len, self.VA), dtype = np.float32)
        
        choice = np.random.randint(0, self.n_films, self.batch_size)
        for i,index in enumerate(choice) :
            
            #Choose random starting point to exploit whole sequences
            start = np.random.random()
            X[i,:,:], Y[i,:,:] = read_data.get_window(index, self.seq_len, start)
            #Model predicts one step ahead of the sequence
            
        return X,Y
        
        
if __name__=="__main__":
    my_dataset = Dataset(3, 150, 3)
    X, Y = my_dataset.get_random_training_batch()
    print(X.shape, Y.shape)
