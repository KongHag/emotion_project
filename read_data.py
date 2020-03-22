# coding:utf-8

"""Extract all the features and valence/arousal

To load the data :
    // Having the file data/x_train.pickle and data/y_train.pickle
    from read_data import load_data
    X, Y = load_data()

Errors found in the data:
    Movie 6 :
        audio feature for frame 5588 is missing

    Movie 18 :
        visual feature :
            fc6 : missing data
            jcd : missing data

    Movie 27 :
        visual feature :
            cl : frame 00006 empty file
    
    Movie 46 :
        valence/arousal 100 values, but the film has around 3000 frames
"""

import numpy as np
import pickle
from log import setup_custom_logger


logger = setup_custom_logger("Read data")

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


def all_audio_features(movie_id):
    """Returns a tensor of the feature values; first dim is the time"""

    image_id = 1
    list_data = []
    while True:
        try:
            list_data.append(audio_features_image_id_to_list(
                movie_id, image_id))
            image_id += 1
        except FileNotFoundError:
            return np.array(list_data, dtype=np.float16)


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
            return np.array(list_data, dtype=np.float16)


def all_visual_features(movie_id):
    """Get all features for given movie"""
    list_data = [visual_feature(movie_id, visual_feature_name)
                 for visual_feature_name in visual_feature_names]
    return np.concatenate(list_data, axis=1)


def valence_arousal(movie_id):
    """Returns a tensor of valence and arousal. First dim is the time"""

    path = valence_arousal_make_path(movie_id)
    with open(path, "r") as f:
        table = [line.split("\t") for line in f.readlines()]
        return np.array([[float(val) for val in row[1:]] for row in table[1:]], dtype=np.float16)


def all_features(movie_id):
    """Returns the concatenation of visual and audio features for a movie

    The numbers of frames are not egal for the visual and the audio feature.
    The overnumerous frame are deleted.
    """
    T_v = all_visual_features(movie_id)
    T_a = all_audio_features(movie_id)
    min_ = min(T_v.shape[0], T_a.shape[0])
    T_v = T_v[:min_, :]
    T_a = T_a[:min_, :]
    return np.concatenate((T_v, T_a), axis=1)


def get_window(movie_id, seq_len, start):
    T = all_features(movie_id)
    VA_T = valence_arousal(movie_id)
    min_ = min(T.shape[0], VA_T.shape[0])
    T = T[:min_, :]
    VA_T = VA_T[:min_, :]
    starting_index = int(start*(T.shape[0]-seq_len+1))
    return T[starting_index:starting_index + seq_len, :], VA_T[starting_index:starting_index + seq_len, :]


def dump_data():
    """ Create a pickle of all the data
    [
        (X_movie0, Y_movie_0),
        (X_movie1, Y_movie_1),
        ...
    ]
    """
    XX = []
    YY = []
    for movie_id in range(66):
        if movie_id in [6, 18, 27, 46]:
            continue
        X = all_features(movie_id)

        logger.info(' '.join(("FEATURES\tmovie :", movie_id,
                              "\tArray shape :", X.shape)))

        Y = valence_arousal(movie_id)
        logger.info(' '.join(("OUTPUT\tmovie :", movie_id,
                              "\tArray shape :", Y.shape)))

        XX.append(X)
        YY.append(Y)

    mean, std = compute_mean_std(XX)
    pickle.dump(mean, open("data/x_mean.pickle", "wb"))
    pickle.dump(std, open("data/x_std.pickle", "wb"))

    pickle.dump(XX, open("data/x_train.pickle", "wb"))
    pickle.dump(YY, open("data/y_train.pickle", "wb"))


def compute_mean_std(all_X):
    """Return an approximate mean and std for every feature"""
    concatenated = np.concatenate(all_X, axis=0).astype(np.float64)

    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    std[std == 0] = 1
    return mean, std


def load_data():
    return(
        pickle.load(open("data/x_train.pickle", "rb")),
        pickle.load(open("data/y_train.pickle", "rb")))


def load_mean_std():
    return(
        pickle.load(open("data/x_mean.pickle", "rb")),
        pickle.load(open("data/x_std.pickle", "rb")))


if __name__ == '__main__':
    X, Y = load_data()
    mean, std = load_mean_std()

    for i, (x, y) in enumerate(zip(X, Y)):
        print(i, len(x), len(y))
