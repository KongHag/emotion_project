# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:42:18 2020

@author: Tim

Usage :
>>> trainset = EmotionDataset()
>>> X, Y = trainset.get_random_training_batch(batch_size=5, seq_len=10)
>>> print("X \ttype :", type(X), "\tshape :", X.shape)
X       type : <class 'numpy.ndarray'>  shape : (5, 10, 6950)
>>> print("Y \ttype :", type(Y), "\tshape :", Y.shape)
Y       type : <class 'numpy.ndarray'>  shape : (5, 10, 2)
"""

import torch
import torch.nn as nn
import numpy as np
from read_data import load_data
from torch.utils.data import Dataset, DataLoader
from log import setup_custom_logger

logger = setup_custom_logger("dataset")
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
        return X, Y


class MediaEval18(Dataset):
    def __init__(self, root='./data', train=True, seq_len=100, shuffle=False, nb_sequences=None):
        self.root = root
        self.train = train
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.nb_sequences = nb_sequences
        self._data_to_sequences_list()

    def _data_to_sequences_list(self):
        if self.train:
            self.data = load_data()[0][:50], load_data()[1][:50]
        else:
            self.data = load_data()[0][50:62], load_data()[1][50:62]

        self._possible_starts = list(self._compute_possible_starts())
        if self.shuffle:
            np.random.shuffle(self._possible_starts)

        if self.nb_sequences is not None:
            self._possible_starts = self._possible_starts[:self.nb_sequences]

    def _compute_possible_starts(self):
        for movie_id, (movie_features, movie_VA) in enumerate(zip(*self.data)):
            duration = min(movie_features.shape[0], movie_VA.shape[0])
            start_idx = 0
            while start_idx + self.seq_len < duration:
                yield {"id_movie": movie_id, "start_idx": start_idx}
                start_idx += 1

    def get_window(self, movie_id, seq_len, start_idx):
        """From a movie_id, returns a sequence of seq_len seconds, starting
        from start (percent) of the all movie"""
        list_movies_features, list_movies_VA = self.data
        movie_features = list_movies_features[movie_id]
        movie_VA = list_movies_VA[movie_id]

        # Tronque (par la fin) la liste la plus longue
        shorter_duration = min(movie_features.shape[0], movie_VA.shape[0])
        movie_features = movie_features[:shorter_duration, :]
        movie_VA = movie_VA[:shorter_duration, :]

        return (movie_features[start_idx:start_idx+seq_len, :],
                movie_VA[start_idx:start_idx+seq_len, :])

    def __len__(self):
        return len(self._possible_starts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        start = self._possible_starts[idx]
        return self.get_window(movie_id=start["id_movie"],
                               seq_len=self.seq_len,
                               start_idx=start["start_idx"])


if __name__ == "__main__":
    # trainset = EmotionDataset()
    import time
    # start = time.time()
    # X, Y = trainset.get_random_training_batch(batch_size=5, seq_len=10)
    # print("X \ttype :", type(X), "\tshape :", X.shape)
    # print("Y \ttype :", type(Y), "\tshape :", Y.shape)
    # print("Batch building duration :\t%.2f" % (time.time() - start))

    trainset = MediaEval18(root='./data', train=True, seq_len=100, nb_sequences=99)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=10, shuffle=True)

    for i, data in enumerate(trainloader):
        print(i, data[0].shape)

