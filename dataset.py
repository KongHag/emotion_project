# -*- coding: utf-8 -*-
"""
Defines the dataset class MediaEval18.

To define a loader that loads shuffled batches of 128 sequences of 20 frames,
described by visual features only, based on 5% of the whole train dataset:
>>> trainset = MediaEval18(root='./data', train=True, seq_len=20,
                           features=['visual'], fragment=0.05)
>>> trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)
>>> for i, data in enumerate(trainloader):
...     print("Size of the batch", i, ":", data[0].shape)
Size of the batch 0 : torch.Size([128, 20, 5367])
Size of the batch 1 : torch.Size([128, 20, 5367])
Size of the batch 2 : torch.Size([128, 20, 5367])
Size of the batch 3 : torch.Size([128, 20, 5367])
...
"""

import torch
import torch.nn as nn
import numpy as np
from read_data import load_data
from torch.utils.data import Dataset, DataLoader
from log import setup_custom_logger

logger = setup_custom_logger("dataset")


class MediaEval18(Dataset):
    """Mediaeval dataset
        root (str, otpional) : default './data'. Path to data pickle
        train (bool, optional) : default True. trainset if True, testset otherwise
        seq_len (int, seq_len) : default 100. nb of frame in a single sample
        shuffle (bool, optional) : default False. Shuffle or not the data
        fragment (float, optional) : default 1. From 0 to 1, percent of dataset used
    """
    _features_len = {
        "acc": range(0, 256),
        "cedd": range(256, 400),
        "cl": range(400, 433),
        "eh": range(433, 513),
        "fcth": range(513, 705),
        "gabor": range(705, 765),
        "jcd": range(765, 933),
        "sc": range(933, 997),
        "tamura": range(997, 1015),
        "lbp": range(1015, 1271),
        "fc6": range(1271, 5367),
        "visual": range(0, 5367),
        "audio": range(5367, 6950),
        "all": range(0, 6950)
    }

    def __init__(self, root='./data', train=True, seq_len=100, shuffle=False,
                 fragment=1, features=['all']):
        self.root = root
        self.train = train
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.features = features

        all_X, all_Y = load_data()
        start_test = 50
        if self.train:
            self.data = all_X[:start_test], all_Y[:start_test]
        else:
            self.data = all_X[start_test:62], all_Y[start_test:62]

        self._possible_starts = list(self._compute_possible_starts())
        if self.shuffle:
            np.random.shuffle(self._possible_starts)

        nb_sequences = int(len(self._possible_starts)*fragment)
        self._possible_starts = self._possible_starts[:nb_sequences]

    def _compute_possible_starts(self):
        for movie_id, (movie_features, movie_VA) in enumerate(zip(*self.data)):
            duration = min(movie_features.shape[0], movie_VA.shape[0])
            start_idx = 0
            while start_idx + self.seq_len < duration:
                yield {"id_movie": movie_id, "start_idx": start_idx}
                start_idx += 1

    def get_img(self, movie_id, img_idx):
        """From a movie_id, returns a sequence of seq_len seconds, starting
        from start (percent) of the all movie"""
        list_movies_features, list_movies_VA = self.data
        movie_features = list_movies_features[movie_id]
        movie_VA = list_movies_VA[movie_id]

        return (movie_features[img_idx], movie_VA[img_idx])

    def _select_features(self, X):
        all_idxs = np.array([], dtype=np.int64)
        for feature_name, idxs in self._features_len.items():
            if feature_name in self.features:
                all_idxs = np.union1d(all_idxs, idxs)
        return X[:, all_idxs]

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
        start = self._possible_starts[idx]
        X, Y = self.get_window(movie_id=start["id_movie"],
                               seq_len=self.seq_len,
                               start_idx=start["start_idx"])
        X = self._select_features(X)
        return X.astype("float32"), Y.astype("float32")


if __name__ == "__main__":
    trainset = MediaEval18(root='./data', train=True, seq_len=20,
                           features=['visual'], fragment=0.05)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)

    for i, data in enumerate(trainloader):
        print("Size of the batch", i, ":", data[0].shape)
