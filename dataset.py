# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:42:18 2020

@author: Tim
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

    def __init__(self, seq_len):
        super(EmotionDataset, self).__init__()
        self.list_movies_features, self.list_movies_VA = load_data()
        # self.x = [
        #     array([[features_t0], [features_t1], ...]),      # movie 0
        #     array([[features_t0], [features_t1], ...]),      # movie 1
        #     ...
        # ]
        # NB : Movies don't have the same nb of seconds

        self.seq_len = seq_len

        # TODO : Convert self.list_movies_features and self.list_movies_VA
        # into a nb.ndarray of sequences (no need to be shuffled), exctracted
        # from all movies (use self.get_window()). The duration of every
        # sequence must be self.seq_len. Store the results in self.x and self.y

        self.x = np.empty(1)
        self.y = np.empty(1)


    def get_window(self, movie_id, start, seq_len):
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

    # def get_random_training_batch(self):
    #     X = np.zeros((self.batch_size, self.seq_len,
    #                   self.input_size), dtype=np.float32)
    #     Y = np.zeros((self.batch_size, self.seq_len,
    #                   self.VA), dtype=np.float32)

    #     choice = np.random.randint(0, self.n_films, self.batch_size)
    #     for i, index in enumerate(choice):

    #         # Choose random starting point to exploit whole sequences
    #         start = np.random.random()
    #         X[i, :, :], Y[i, :, :] = read_data.get_window(
    #             index, self.seq_len, start)
    #         # Model predicts one step ahead of the sequence

    #     return X, Y


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


trainset = EmotionDataset(seq_len=200)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)

if __name__ == "__main__":
    import time
    start = time.time()
    list_movies_features, list_movies_VA = load_data()
    print("Loading duration :\t\t%.2f" % (time.time() - start))
    print("Number of films loaded :\t", len(list_movies_features))
