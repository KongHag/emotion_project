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

class MediaEval18(Dataset):
    """Mediaeval dataset
        root (str, otpional) : default './data'. Path to data pickle
        train (bool, optional) : default True. trainset if True, testset otherwise
        seq_len (int, seq_len) : default 100. nb of frame in a single sample
        shuffle (bool, optional) : default False. Shuffle or not the data
        fragment (float, optional) : default 1. From 0 to 1, percent of dataset used
    """
    _features_len = {
        "acc": [0, 256],
        "cedd": [256, 400],
        "cl": [400, 433],
        "eh": [433, 513],
        "fcth": [513, 705],
        "gabor": [705, 765],
        "jcd": [765, 933],
        "sc": [933, 997],
        "tamura": [997, 1015],
        "lbp": [1015, 1271],
        "fc6": [1271, 5367],
        "audio": [5367, 6960]
    }

    def __init__(self, root='./data', train=True, shuffle=False,
                 fragment=1, features=['all']):
        self.root = root
        self.train = train
        self.shuffle = shuffle
        self.features = features

        all_X, all_Y = load_data()
        i=50
        if self.train:
            self.data = all_X[:i], all_Y[:i]
        else:
            self.data = all_X[i:62], all_Y[i:62]

        self._possible_imgs = list(self._compute_possible_imgs())
        if self.shuffle:
            np.random.shuffle(self._possible_imgs)

        nb_imgs=int(len(self._possible_imgs)*fragment)
        self._possible_imgs = self._possible_imgs[:nb_imgs]

    def _compute_possible_imgs(self):
        for movie_id, (movie_features, movie_VA) in enumerate(zip(*self.data)):
            duration = min(movie_features.shape[0], movie_VA.shape[0])
            img_idx = 0
            while img_idx < duration:
                yield {"id_movie": movie_id, "img_idx": img_idx}
                img_idx += 10

    def get_img(self, movie_id, img_idx):
        """From a movie_id, returns a sequence of seq_len seconds, starting
        from start (percent) of the all movie"""
        list_movies_features, list_movies_VA = self.data
        movie_features = list_movies_features[movie_id]
        movie_VA = list_movies_VA[movie_id]

        return (movie_features[img_idx], movie_VA[img_idx])

    def _select_features(self, X):
        new_seq = []
        for feature_name, idxs in self._features_len.items():
            if feature_name in self.features or 'all' in self.features:
                new_seq.append(X[idxs[0]:idxs[1]])
        return np.concatenate(new_seq, axis=0)

    def __len__(self):
        return len(self._possible_imgs)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        img = self._possible_imgs[idx]
        X, Y = self.get_img(movie_id=img["id_movie"],
                            img_idx=img["img_idx"])
        X = self._select_features(X)
        return X.astype("float32"), Y.astype("float32")


if __name__ == "__main__":
    # trainset = EmotionDataset()
    import time
    # start = time.time()
    # X, Y = trainset.get_random_training_batch(batch_size=5, seq_len=10)
    # print("X \ttype :", type(X), "\tshape :", X.shape)
    # print("Y \ttype :", type(Y), "\tshape :", Y.shape)
    # print("Batch building duration :\t%.2f" % (time.time() - start))

    trainset = MediaEval18(root='./data', train=True, features=['fc6'], fragment=0.005)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)

    print(len(trainset))
    for i, data in enumerate(trainloader):
        print(i, data[0].shape)
