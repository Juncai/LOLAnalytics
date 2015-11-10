import utils.io as io
import Consts as c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


player_dict_path = 'data/player_dict.pickle'

def k_means():
    print('Loading player dict...')
    player_dict = io.load_pickle(player_dict_path)
    # TODO get data from the dict
    print('Getting features from the dict...')
    player_features = np.array([p[c.FEATURES] for p in player_dict.values()])


    # plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170

    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(player_features)

    print(y_pred)

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

if __name__ == '__main__':
    hierarchical()