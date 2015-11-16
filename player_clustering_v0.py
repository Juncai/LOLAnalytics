import utils.io as io
import Consts as c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance


player_dict_path = 'data/player_dict.pickle'

def k_means():
    cluster_num = 5
    print('Loading player dict...')
    player_dict = io.load_pickle(player_dict_path)
    # TODO get data from the dict
    print('Getting features from the dict...')
    player_features = np.array([p[c.FEATURES] for p in player_dict.values()]) / 500


    # plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170

    # Incorrect number of clusters
    km = KMeans(n_clusters=cluster_num, random_state=random_state)
    y_pred = km.fit_predict(player_features)

    evaluate_result = []
    for i in range(cluster_num):
        cur_f = [f for ind, f in enumerate(player_features) if y_pred[ind] == i]
        evaluate_result.append(evaluate_clustering(cur_f, km.cluster_centers_[i]))

    mean_eval = np.mean(evaluate_result)

    print(km.cluster_centers_)
    print(y_pred)
    print(km.get_params())
    print('Mean distance: {}'.format(mean_eval))
    # plt.subplot(221)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.title("Incorrect Number of Blobs")


    # plt.show()

def hierarchical():
    print('Loading player dict...')
    player_dict = io.load_pickle(player_dict_path)
    # TODO get data from the dict
    print('Getting features from the dict...')
    player_features = np.array([p[c.FEATURES] for p in player_dict.values()])


    # plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170

    for linkage in ('ward', 'average', 'complete'):
        y_pred = AgglomerativeClustering(linkage=linkage, n_clusters=5).fit_predict(player_features)
        print('Linkage type: {}'.format(linkage))
        print(y_pred)

    # plt.subplot(221)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.title("Incorrect Number of Blobs")


    # plt.show()

def evaluate_clustering(features, center):
    n = len(features)
    return distance.cdist(features, center)


if __name__ == '__main__':
    # hierarchical()
    # k_means()
    f = [(1, 2), (1, 2)]
    c = [[0, 0], [1, 3]]
    print(evaluate_clustering(f, c))
