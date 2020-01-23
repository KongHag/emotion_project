# coding:utf-8

"""Extract all the visual feature

Erreur trouvée dans les données:
    MEDIAEVAL18_27-00006_cl.txt     : empty
    Film 18 feature fc6 : données lacunaires
    Film 18 feature jcd : données lacunaires
"""

import torch

features = ["acc", "cedd", "cl", "eh", "fcth", "gabor", "jcd",
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


def make_path(movie_id, feature, image_id):
    """Returns the path of the corresponding file

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


def image_id_to_list(movie_id, feature, image_id):
    """Returns a list of float regarding the movie, the feature and the image_id"""

    path = make_path(movie_id, feature, image_id)
    with open(path, "r") as f:
        return [float(val) for val in f.readline().split(",")]


def movie_id_feature_to_tensor(movie_id, feature):
    """Returns a tensor of the descrptor values; first dim is the time"""

    image_id = 1
    list_data = []
    while True:
        try:
            list_data.append(image_id_to_list(movie_id, feature, image_id))
            image_id += 1
        except FileNotFoundError:
            return torch.tensor(list_data)


if __name__ == '__main__':
    for feature in features:
        for movie_id in range(66):
            try:
                my_tensor = movie_id_feature_to_tensor(movie_id, feature)
                print("movie :", movie_id, "Feature :", feature,
                      "Tensor shape :", my_tensor.shape)
            except Exception as exc:
                print("Film {}, feature {} : erreur de lecture".format(
                    movie_id, feature))
