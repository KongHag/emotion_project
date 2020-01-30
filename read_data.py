# coding:utf-8

"""Extract all the visual feature

Erreur trouvée dans les données:
    MEDIAEVAL18_27-00006_cl.txt     : empty
    Film 18 feature fc6 : données lacunaires
    Film 18 feature jcd : données lacunaires

    Film 6 frame 5588 : pas de données
"""

import torch
import arff
import numpy as np

visual_feature_names = ["acc", "cedd", "cl", "eh", "fcth", "gabor", "jcd",
                        "sc", "tamura", "lbp", "fc6"]


def movie_id_to_set_name(movie_id):
    """Returns the the set name of the movie"""
    if movie_id < 14:
        return "DevSet-Part1"
    elif movie_id < 44:
        return "DevSet-Part2"
    elif movie_id < 54:
        return "DevSet-Part3"
    else:
        return "TestSet"


def visual_features_make_path(movie_id, feature, image_id):
    """Returns the path of the corresponding file for the visual features

    movie_id (int) : from 0 to 65
    feature (str): in ["acc", "cedd", "cl", "eh", "fcth", "gabor", "jcd",
        "sc", "tamura", "lbp", "fc6"]
    image_id (int) : number of seconds from the begining of the movie
    """

    # Convert id into string
    str_movie_id = "%02d" % movie_id  # 1 becomes "01"
    str_image_id = "%05d" % image_id  # 13 becomes "00013"

    set_name = movie_id_to_set_name(movie_id)
    path = ("data/MEDIAEVAL18-{0}-Visual_features/visual_features"
            + "/MEDIAEVAL18_{1}/{2}/MEDIAEVAL18_{1}-{3}_{2}.txt")

    return path.format(set_name, str_movie_id, feature, str_image_id)


def valence_arousal_make_path(movie_id):
    """Returns the path of the corresponding file for valence/arousal

    movie_id (int) : from 0 to 65
    """

    # Convert id into string
    str_movie_id = "%02d" % movie_id  # 1 becomes "01"

    set_name = movie_id_to_set_name(movie_id)
    path = ("data/MEDIAEVAL18-{}-Valence_Arousal-annotations/annotations"
            + "/MEDIAEVAL18_{}_Valence-Arousal.txt")

    return path.format(set_name, str_movie_id)


def audio_features_make_path(movie_id, image_id):
    """Returns the path of the corresponding file for the audio features

    movie_id (int) : from 0 to 65
    image_id (int) : number of seconds from the begining of the movie
    """

    # Convert id into string
    str_movie_id = "%02d" % movie_id  # 1 becomes "01"
    str_image_id = "%05d" % image_id  # 13 becomes "00013"

    set_name = movie_id_to_set_name(movie_id)
    path = ("data/MEDIAEVAL18-{0}-Audio_features/audio_features"
            + "/MEDIAEVAL18_{1}/MEDIAEVAL18_{1}_{2}.csv")

    return path.format(set_name, str_movie_id, str_image_id)


def visual_features_image_id_to_list(movie_id, feature, image_id):
    """Returns a list of float regarding the movie, the feature and the image_id"""

    path = visual_features_make_path(movie_id, feature, image_id)
    with open(path, "r") as f:
        return [float(val) for val in f.readline().split(",")]


def audio_features_image_id_to_list(movie_id, image_id):
    """Returns the list of float regarding the movie, and the image_id"""

    path = audio_features_make_path(movie_id, image_id)
    _, iterator_val = arff.load(path)

    audio_features_values = []
    for i, val in enumerate(iterator_val):
        if i == 0:
            continue
        audio_features_values.append(val)

    return audio_features_values


def all_audio_feature(movie_id):
    """Returns a tensor of the feature values; first dim is the time

    movie_id (int): the id of the movie
    """

    image_id = 1
    list_data = []
    while True:
        try:
            list_data.append(audio_features_image_id_to_list(
                movie_id, image_id))
            image_id += 1
        except FileNotFoundError:
            return torch.tensor(list_data)


def visual_feature(movie_id, feature):
    """Returns a tensor of the feature values; first dim is the time

    movie_id (int): the id of the movie
    feature (str) : in ["acc", "cedd", "cl", "eh", "fcth", "gabor", "jcd",
        "sc", "tamura", "lbp", "fc6"]
    """

    image_id = 1
    list_data = []
    while True:
        try:
            list_data.append(visual_features_image_id_to_list(
                movie_id, feature, image_id))
            image_id += 1
        except FileNotFoundError:
            return torch.tensor(list_data)


def all_visual_feature(movie_id):
    """Get all descriptors for given movie"""
    list_data = []
    for f in visual_feature_names:
        list_data.append(visual_feature(movie_id, f))
    return torch.cat(list_data, dim=1)


def valence_arousal(movie_id):
    """Returns a tensor of valence and arousal. First dim is the time"""

    path = valence_arousal_make_path(movie_id)
    with open(path, "r") as f:
        table = [line.split("\t") for line in f.readlines()]
        return torch.tensor([[float(val) for val in row[1:]] for row in table[1:]])


def all_features(movie_id):
    T_v = all_visual_feature(movie_id)
    T_a = all_audio_feature(movie_id)
    min_ = min(T_v.shape[0], T_a.shape[0])
    T_v = T_v[:min_,:]
    T_a = T_a[:min_,:]
    return torch.cat([T_v,T_a], dim = 1)



def get_window(movie_id, seq_len, start):
    T = all_features(movie_id)
    VA_T = valence_arousal(movie_id)
    min_ = min(T.shape[0], VA_T.shape[0])
    T = T[:min_,:]
    VA_T = VA_T[:min_,:]
    starting_index = start*(T.shape[0]-seq_len+1)
    return T[starting_index:starting_index + seq_len,:], VA_T[starting_index:starting_index + seq_len,:]


if __name__ == '__main__':
    for movie_id in range(1, 4):
        if movie_id in [6, 18]:
            continue

        my_tensor = valence_arousal(movie_id)
        print("VALENCE/AROUSAL\tmovie :", movie_id,
            "\tTensor shape :", my_tensor.shape)

        my_tensor = all_visual_feature(movie_id,)
        print("VISUAL FEATURE\tmovie :", movie_id,
            "\tTensor shape :", my_tensor.shape)

        my_tensor = all_audio_feature(movie_id)
        print("AUDIO FEATURE\tmovie :", movie_id,
            "\tTensor shape :", my_tensor.shape)
        
            
