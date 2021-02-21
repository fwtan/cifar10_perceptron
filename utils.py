import pickle
import numpy as np
import os.path as osp


def pickle_load(path):
    with open(path, 'rb') as fid:
        try:
            data_ = pickle.load(fid)
        except:
            fid.seek(0)
            data_ = pickle.load(fid, encoding='latin1')
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def load_data(data_dir):
    Raw_X_train, Raw_y_train, Raw_X_test, Raw_y_test = load_CIFAR10(data_dir)

    train_inds = np.array([int(i) for i in read_file(osp.join(data_dir, 'train.txt'))])
    val_inds   = np.array([int(i) for i in read_file(osp.join(data_dir, 'val.txt'))])
    test_inds  = np.array([int(i) for i in read_file(osp.join(data_dir, 'test.txt'))])

    X_train = Raw_X_train[train_inds]
    y_train = Raw_y_train[train_inds]

    X_val   = Raw_X_train[val_inds]
    y_val   = Raw_y_train[val_inds]

    X_test = Raw_X_test[test_inds]
    y_test = Raw_y_test[test_inds]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val   -= mean_image
    X_test  -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(len(train_inds), -1)
    X_val   = X_val.reshape(len(val_inds), -1)
    X_test  = X_test.reshape(len(test_inds), -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


####################################################################
## Code borrowed from https://github.com/cs231n/cs231n.github.io

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    datadict = pickle_load(filename)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y 


def load_CIFAR10(ROOT):
    xs, ys = [], []
    for b in range(1,6):
        f = osp.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    Xtest, Ytest = load_CIFAR_batch(osp.join(ROOT, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest
