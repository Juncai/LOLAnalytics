import Consts as c
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import distance
import numbers
import Utils as util
import Preprocess
from sklearn.linear_model import LogisticRegression
import time
from sklearn.preprocessing import normalize
from sklearn import manifold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# player_dict_path = 'data/player_dict.pickle'
player_dict_path = 'data/player_dict_3.pickle'  # 10770 players with 300 matches each
perfect_match_path = 'data/perfect_matches.pickle'

def main():

    # centers = [i for i in range(3, 15)]
    #
    # for c in centers:
    #     k_means(c)
    st = time.time()
    n_center = 6
    cluster_method = 'hierarchical'

    # load match data
    print('{} Loading match data...'.format(time.time() - st))
    match_dict = util.load_pickle_file(perfect_match_path)


    print('Loading player dict...')
    player_dict = util.load_pickle(player_dict_path)

    # get data from the dict
    print('Getting features from the dict...')

    player_features_id = np.array([np.append(player_dict[pid][c.FEATURES], pid) for pid in player_dict])  # last column is pid
    player_features = player_features_id[:, 0 : -1]

    n = len(player_features)


    # 2D embedding of the digits dataset
    print("Computing embedding")
    player_features = manifold.SpectralEmbedding(n_components=3).fit_transform(player_features)
    print("Done.")


    for cluster_method in ('kmeans', 'hierarchical'):
        for n_center in range(2, 10):
            print('Clustering: {}, number of clusters: {}: '.format(cluster_method, n_center))
            # clustering and get the distance to center as the new features
            print('{} Clustering...'.format(time.time() - st))
            new_features = cluster(cluster_method, n_center, player_features)

            player_feature_dict = {}
            for i in range(n):
                player_f = new_features[i]
                player_feature_dict[player_features_id[i][-1]] = player_f


            # construct new features as a team play style (currently a simple aggregation of all the players' play style)
            print('{} Constructing new dataset...'.format(time.time() - st))

            features = []
            label = []
            flip = False    # flag for flip win/lose every match
            for mid in match_dict:
                m = match_dict[mid]
                win_f = np.zeros((1, n_center))[0]
                lose_f = np.zeros((1, n_center))[0]
                for pid in m[0][c.TEAM_INFO_PLAYERS]:
                    win_f += player_feature_dict[pid]
                for pid in m[1][c.TEAM_INFO_PLAYERS]:
                    lose_f += player_feature_dict[pid]
                win_f /= 5
                lose_f /= 5
                if flip:
                    features.append(np.append(lose_f, win_f))
                    label.append(-1)
                else:
                    features.append(np.append(win_f, lose_f))
                    label.append(1)
                flip = not flip  # flip the flag

            features = np.array(features)
            label = np.array(label)

            # normalize features
            # row_ranges = features.max(axis=1) - features.min(axis = 1)
            # row_means = features.mean(axis=1)
            # features = features / row_means[:, np.newaxis]
            # features = normalize(features)



            # prepare training and testing set
            print('{} Start training...'.format(time.time() - st))
            k = 9
            k_folds = Preprocess.prepare_k_folds([features, label], k)

            for i in range(k):
                tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
                tr_n, f_d = np.shape(tr_data[0])
                te_n, = np.shape(te_data[1])

                # train with some algorithm

                # clf1 = LogisticRegression(random_state=123)

                clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

                clf1.fit(tr_data[0], tr_data[1])
                tr_pred1 = clf1.predict(tr_data[0])
                te_pred1 = clf1.predict(te_data[0])
                tr_acc = (tr_pred1 == tr_data[1]).sum() / tr_n
                te_acc = (te_pred1 == te_data[1]).sum() / te_n
                print('Training acc: {}, Testing acc: {}'.format(tr_acc, te_acc))




def cluster(method, center_num, player_features, linkage='ward'):
    '''

    :param center_num:
    :return: predictions, centers, distance to each center
    '''
    # if isinstance(center_num, numbers.Number):
    #     pass



    y_pred = []
    clustering = None
    if method == 'kmeans':
        # random_state = 170

        # km = KMeans(n_clusters=center_num, random_state=random_state)
        clustering = KMeans(n_clusters=center_num)
    elif method == 'hierarchical':
        # for linkage in ('ward', 'average', 'complete'):
        linkage = 'ward'
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=center_num)

    new_features = clustering.fit_predict(player_features)

    if method == 'kmeans':
        # calculate distance between all datapoint and center pairs
        # new_features = dist_to_center(player_features, km.cluster_centers_)
        new_features = np.dot(player_features, np.transpose(clustering.cluster_centers_))

    return new_features

def hierarchical():
    print('Loading player dict...')
    player_dict = util.load_pickle(player_dict_path)
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

def dist_to_center(features, centers):
    # res = distance.cdist(features, centers)
    # res = distance.cdist(features, centers, 'minkowsk')
    # res = distance.cdist(features, centers, 'correlation')
    # res = distance.cdist(features, centers, 'chebyshev')
    res = distance.cdist(features, centers, 'seuclidean')
    return res


if __name__ == '__main__':
    # hierarchical()
    # k_means()
    # f = [(1, 2), (1, 2), (3, 3)]
    # c = [[0, 0], [1, 3]]
    # print(dist_to_center(f, c))
    main()
