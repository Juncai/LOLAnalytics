import pickle

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

if __name__ == '__main__':
    load_features()