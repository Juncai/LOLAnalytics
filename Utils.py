import pickle
import numpy as np
import Consts as c

def load_pickle(path, type=None):
    res = None
    with open(path, 'rb') as f:
        res = pickle.load(f)
    if type is not None:
        assert isinstance(res, type)
    return res

def load_features():
    feature_file_path = '../data/player_features_v1'

    with open(feature_file_path, 'r') as f:
        for line in f:
            line = line[:-1]
            # print("{} = '{}'".format(camel_to_ud_split(line), line))
            print(camel_to_ud_split(line))

def camel_to_ud_split(s):
    return ''.join(c.upper() if c.islower() else ('_' + c) for c in s)


def load_pickle_file(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f)

def convert_feature_to_nparray(path):
    p_dict = load_pickle_file(path)
    for p in p_dict.keys():
        p_dict[p][c.FEATURES] = np.array(p_dict[p][c.FEATURES])
    save(p_dict, path)


if __name__ == '__main__':
    p_dict_path = 'data/player_dict.pickle'
    convert_feature_to_nparray(p_dict_path)
