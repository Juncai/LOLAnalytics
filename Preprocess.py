import numpy as np
import numpy.random as random
import pickle


def generate_thresholds(features, thresh_path):
    threshs = []
    for i in range(len(features[0])):
        cur_f = [x[i] for x in features]
        threshs.append(np.unique(cur_f))
    f = open(thresh_path, 'wb+')
    pickle.dump(threshs, f)
    f.close()
    return threshs


def shift_and_scale(ds, col):
    '''
    normalize all features in the given dataset with shift and scale
    method
    '''
    tmp_ar = [x[col] for x in ds]
    min_val = np.amin(tmp_ar)
    max_val = np.amax(tmp_ar)
    range_val = max_val - min_val
    for i in range(len(tmp_ar)):
        ds[i][col] = (tmp_ar[i] - min_val) / range_val


def zero_mean_unit_var(ds, col):
    '''
    normalize all features in the given dataset with zero mean
    and unit variance method
    '''
    tmp_ar = [x[col] for x in ds]
    mean_val = np.mean(tmp_ar)
    std_val = np.std(tmp_ar)
    for i in range(len(tmp_ar)):
        ds[i][col] = (tmp_ar[i] - mean_val) / std_val


def normalize_features(methods, cols, train_ds, test_ds=None):
    '''
    normalize the given features with specified method for each col
    '''
    cur_ds = train_ds
    if test_ds:
        cur_ds = train_ds + test_ds

    for i, m in enumerate(methods):
        m(cur_ds, cols[i])
    if test_ds:
        train_ds = cur_ds[0:len(train_ds)]
        test_ds = cur_ds[len(train_ds):]


def normalize_features_all(method, train_ds, test_ds=None, not_norm=()):
    '''
    Apply given normalize method to all of the feature columns
    '''
    if method is None:
        return
    cur_ds = train_ds
    if test_ds:
        cur_ds = train_ds + test_ds


    for i in range(len(cur_ds[0])):    # data only contain features
        if i not in not_norm:
            method(cur_ds, i)

    if test_ds:
        train_ds = cur_ds[0:len(train_ds)]
        test_ds = cur_ds[len(train_ds):]

# def preprocess_data(path, test_path=None):
#     (label, features) = file_to_dataset(path)
#     if test_path:
#         (label_test, features_test) = file_to_dataset(test_path)


def prepare_k_fold_data(dataset, k, ind):
    '''
    Prepare K-Fold training and testing data
    '''
    features = dataset[0]
    label = dataset[1]
    count = len(label)
    training_f = []
    training_l = []
    testing_f = []
    testing_l = []

    p1 = int(1.0 * (ind - 1) * count / k)
    p2 = int(1.0 * ind * count / k)
    training_f += features[0:p1] + features[p2:]
    training_l += label[0:p1] + label[p2:]
    testing_f += features[p1:p2]
    testing_l += label[p1:p2]
    return ((training_f, training_l), (testing_f, testing_l))


def prepare_k_folds(dataset, k):
    features = dataset[0]
    label = dataset[1]
    cnt = len(label)
    k_folds = []

    for i in range(k):
        i_fold = [[], []]
        j = i
        while j < cnt:
            i_fold[0].append(features[j])
            i_fold[1].append(label[j])
            j += k
        k_folds.append(i_fold)
    return k_folds


def get_i_fold(k_folds, ind):
    '''
    Get ith fold as testing data, the rest be the training data
    :param k_folds:
    :param ind:
    :return:
    '''
    testing_data = k_folds[ind]
    training_data = [[], []]

    for i, f in enumerate(k_folds):
        if i != ind:
            training_data[0] += f[0]
            training_data[1] += f[1]
    return training_data, testing_data

def get_c_percent(c, ds):
    '''
    Use ith fold as testing data, then get c percent of the rest to be training data
    :param k_folds:
    :param ind:
    :return:
    '''
    n = len(ds[0])
    size = int(n * c / 100)
    all = [i for i in range(n)]
    tr = random.choice(all, size, replace=False)
    res = ([], [])
    for i in tr:
        res[0].append(ds[0][i])
        res[1].append(ds[1][i])
    return  res
