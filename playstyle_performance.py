import player_clustering_v0 as pc



player_dict_path = 'data/player_dict.pickle'


def main():
    center_num = 5
    _, centers = pc.k_means(center_num)
    player_dict = io.load_pickle(player_dict_path)
    feature_dict = get_dist_as_features(player_dict, centers)

def get_dist_as_features(p_dict, centers):
    res = {}
    for k in p_dict:
        res[k] =


