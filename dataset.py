# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:42:18 2020

@author: Tim

Usage :
>>> trainset = EmotionDataset()
>>> X, Y = trainset.get_random_training_batch(batch_size=5, seq_len=10)

"""

import torch
import torch.nn as nn
import numpy as np
from read_data import load_data

from torch.utils.data import Dataset, DataLoader

# %%


class EmotionDataset(Dataset):
    """The dataset:
    1 item is a sequence of seq_len seconds. The dim of the tensor are
    (time, features)
    """

    def __init__(self):
        super(EmotionDataset, self).__init__()
        self.list_movies_features, self.list_movies_VA = load_data()
        # self.x = [
        #     array([[features_t0], [features_t1], ...]),      # movie 0
        #     array([[features_t0], [features_t1], ...]),      # movie 1
        #     ...
        # ]
        # NB : Movies don't have the same nb of seconds


        # Number of features
        self.input_size = self.list_movies_features[0].shape[1]

        # self.dim_output = 2 (=len([valence, arousal]))
        self.output_size = self.list_movies_VA[0].shape[1]


    def get_window(self, movie_id, seq_len, start):
        """From a movie_id, returns a sequence of seq_len seconds, starting
        from start (percent) of the all movie"""
        movie_features = self.list_movies_features[movie_id]
        movie_VA = self.list_movies_VA[movie_id]

        # Tronque (par la fin) la liste la plus longue
        shorter_duration = min(movie_features.shape[0], movie_VA.shape[0])
        movie_features = movie_features[:shorter_duration, :]
        movie_VA = movie_VA[:shorter_duration, :]

        starting_index = int(start*(shorter_duration-seq_len+1))
        return (movie_features[starting_index:starting_index+seq_len, :],
                movie_VA[starting_index:starting_index+seq_len, :])

    def get_random_training_batch(self, batch_size, seq_len):
        X = np.zeros((batch_size, seq_len,
                      self.input_size), dtype=np.float32)
        Y = np.zeros((batch_size, seq_len,
                      self.output_size), dtype=np.float32)

        random_movie_ids = np.random.randint(
            0, len(self.list_movies_features), batch_size)

        for i, movie_id in enumerate(random_movie_ids):
            # Choose random starting point to exploit whole sequences
            start = np.random.random()
            X[i, :, :], Y[i, :, :] = self.get_window(movie_id, seq_len, start)
            # Model predicts one step ahead of the sequence
        return torch.from_numpy(X), torch.from_numpy(Y)

    # built-in method are useless since we don't use a dataloader
    # def __len__(self):
    #     return len(self.x)

    # def __getitem__(self, idx):
    #     return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])




if __name__ == "__main__":
    trainset = EmotionDataset()
    import time
    start = time.time()
    X, Y = trainset.get_random_training_batch(batch_size=50, seq_len=100)
    print("Batch building duration :\t%.2f" % (time.time() - start))