import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle
import os

from Timer import timer, ticker


def prepare_data(faces_data_file_name, not_face_data_file_name):
    faces = np.load(faces_data_file_name)
    not_faces = np.load(not_face_data_file_name)

    X = np.vstack((faces, not_faces))
    y = np.hstack((np.ones(faces.shape[0]), -np.ones(not_faces.shape[0])))
    w = np.hstack((np.ones(faces.shape[0])/faces.shape[0]/2, np.ones(not_faces.shape[0])/not_faces.shape[0]/2))

    np.random.seed(0)
    permutation = np.array([i for i in range(y.shape[0])])
    np.random.shuffle(permutation)

    for i in range(len(permutation)):
        pi = permutation[i]
        X[i], X[pi] = X[pi], X[i]
        y[i], y[pi] = y[pi], y[i]
        w[i], w[pi] = w[pi], w[i]

    return X, y, w


def train():
    timer.start()
    print('started preparing data')
    X, y, w = prepare_data('features_full/train/face/data.npy', 'features_full/train/non-face/data.npy')
    print('done preparing data', str(timer.stop()) + str('s'))

    ada_boost = AdaBoostClassifier()
    timer.start()
    print('started training')
    ada_boost.fit(X, y, w)
    print('done training', str(timer.stop()) + str('s'))

    with open('ada_boosts/wighted.pickle', 'wb') as handle:
        pickle.dump(ada_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_model(model_path):
    with open('ada_boosts/' + model_path, 'rb') as handle:
        ada_boost = pickle.load(handle)

    faces_folder = 'features_partial/test/face/'
    non_faces_folder = 'features_partial/test/non-face/'

    timer.start()
    tick = ticker.start_track(5, lambda: print(int((tp + fn)/len(os.listdir(faces_folder))*100), '%'))

    tp = 0
    fn = 0
    for filename in os.listdir(faces_folder):
        feature_vector = np.load(faces_folder + filename)
        feature_vector = np.asmatrix(feature_vector)

        pred = ada_boost.predict(feature_vector)

        if pred == -1:
            fn += 1
        else:
            tp += 1
    print('positive processed in', timer.stop())

    ticker.stop_track(tick)

    timer.start()
    tick = ticker.start_track(5, lambda: print(int((tn + fp)/len(os.listdir(non_faces_folder))*100), '%'))

    tn = 0
    fp = 0
    for filename in os.listdir(non_faces_folder):
        feature_vector = np.load(non_faces_folder + filename)
        feature_vector = np.asmatrix(feature_vector)

        pred = ada_boost.predict(feature_vector)

        if pred == -1:
            tn += 1
        else:
            fp += 1

    print('negative processed in', timer.stop())
    ticker.stop_track(tick)

    N = tp + tn + fp + fn

    print('accuracy: ', (tp + tn)/N)
    print('type I error: ', fp/(fp + tn))
    print('type II error: ', fn/(tp + fn))


train()
# test_model('regular.pickle')
