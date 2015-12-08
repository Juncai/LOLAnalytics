# with champion info


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
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

# player_dict_path = 'data/player_dict.pickle'
player_dict_path = 'data/player_dict_with_champ.pickle'  # 10770 players with 300 matches each
perfect_match_path = 'data/perfect_matches_with_champ_info.pickle'
champ_tags_path = 'data/champ_tags.pickle'

def main():

    # centers = [i for i in range(3, 15)]
    #
    # for c in centers:
    #     k_means(c)
    st = time.time()

    # load data
    print('{} Loading match data...'.format(time.time() - st))
    match_dict = util.load_pickle_file(perfect_match_path)


    print('Loading player dict...')
    player_dict = util.load_pickle(player_dict_path)

    print('Loading champion tags...')
    champ_tags = util.load_pickle(champ_tags_path)
    champ_tags_list = list(champ_tags[0])
    champ_tags_dict = champ_tags[1]

    # get data from the dict
    print('Getting features from the dict...')
    # each player feature will have six elements (one for each champ tag)
    # ORDER: Tank, Marksman, Support, Fighter, Mage, Assassin
    n_p_feature = 44
    player_feature_dict_pre = {}
    col_to_del = [1, 34, 35]
    for pid in player_dict:
        player_feature_dict_pre[pid] = {c.MATCH_COUNT : np.zeros((6,)), c.FEATURES : []}
        for i in range(6):
            player_feature_dict_pre[pid][c.FEATURES].append(np.zeros((n_p_feature,)))
        for cid in player_dict[pid]:
            for t in champ_tags_dict[cid]:
                cur_f = np.delete(player_dict[pid][cid][c.FEATURES], col_to_del)
                player_feature_dict_pre[pid][c.FEATURES][champ_tags_list.index(t)] += cur_f
                player_feature_dict_pre[pid][c.MATCH_COUNT][champ_tags_list.index(t)] += player_dict[pid][cid][c.MATCH_COUNT]
        for i, f in enumerate(player_feature_dict_pre[pid][c.FEATURES]):
            cur_m_count = player_feature_dict_pre[pid][c.MATCH_COUNT][i]
            f /= (cur_m_count if cur_m_count > 0 else 1)

    # player_features_id = np.array([np.append(player_dict[pid][c.FEATURES], pid) for pid in player_dict])  # last column is pid
    # player_features = player_features_id[:, 0 : -1]




    # n = len(player_features)
    #
    #
    # # 2D embedding of the digits dataset
    # print("Computing embedding")
    # player_features = manifold.SpectralEmbedding(n_components=2).fit_transform(player_features)
    # print("Done.")

    # construct new features as a team play style (currently a simple aggregation of all the players' play style)
    print('{} Constructing new dataset...'.format(time.time() - st))
    n_feature = n_p_feature * len(champ_tags_list)
    features = []
    label = []
    flip = False    # flag for flip win/lose every match
    for mid, m in match_dict.items():
        win_f = np.zeros((n_feature,))
        loss_f = np.zeros((n_feature,))
        team_f = [win_f, loss_f]
        for t_ind, team in enumerate(m):
            ct_count = np.zeros((6,))  # counts for each champion tag
            for ind, pid in enumerate(team[c.TEAM_INFO_PLAYERS]):
                champ_id = team[c.TEAM_INFO_CHAMPIONS][ind]
                champ_tags = champ_tags_dict[champ_id]
                for ct in champ_tags:
                    ct_ind = champ_tags_list.index(ct)
                    ct_count[ct_ind] += 1
                    start_col = 0 + ct_ind * n_p_feature
                    end_col = (ct_ind + 1) * n_p_feature
                    cur_pf = player_feature_dict_pre[pid][c.FEATURES][ct_ind]
                    # print("ct: {}, ct_ind: {}, start_col: {}, end_col: {}".format(ct, ct_ind, start_col, end_col))
                    # print(team_f[t_ind][start_col:end_col])
                    # print(cur_pf)
                    team_f[t_ind][start_col:end_col] += cur_pf
            for ctc_ind, ctc in enumerate(ct_count):
                start_col = 0 + ctc_ind * n_p_feature
                end_col = (ctc_ind + 1) * n_p_feature
                if ctc > 1:
                    team_f[t_ind][start_col:end_col] /= ctc
                elif ctc == 0:
                    for pid in team[c.TEAM_INFO_PLAYERS]:
                        team_f[t_ind][start_col:end_col] +=  player_feature_dict_pre[pid][c.FEATURES][ctc_ind]
                    team_f[t_ind][start_col:end_col] /= 5

        # TODO calculate feature mean
        if np.random.random_sample() >= 0.5:
            # features.append(np.append(loss_f, win_f))
            features.append(loss_f - win_f)
            label.append(-1)
        else:
            # features.append(np.append(win_f, loss_f))
            features.append(win_f - loss_f)
            label.append(1)
        flip = not flip  # flip the flag

    features = np.array(features)
    label = np.array(label)

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

        clf1 = LogisticRegression(random_state=123)
        cc = 0.01
        kernel = 'linear'
        tol = 0.01
        # clf1 = svm.SVC(C=cc, kernel=kernel, tol=tol)
        # clf1 = KNeighborsClassifier(n_neighbors=5)

        # clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
        #          algorithm="SAMME",
        #          n_estimators=200)

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
