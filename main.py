import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle

from Timer import timer


def prepare_data(faces_data_file_name, not_face_data_file_name):
    faces = np.load(faces_data_file_name)
    not_faces = np.load(not_face_data_file_name)

    X = np.vstack((faces, not_faces))
    y = np.hstack((np.ones(faces.shape[0]), -np.ones(not_faces.shape[0])))

    np.random.seed(0)
    permutation = np.array([i for i in range(y.shape[0])])
    np.random.shuffle(permutation)

    for i in range(len(permutation)):
        pi = permutation[i]
        X[i], X[pi] = X[pi], X[i]
        y[i], y[pi] = y[pi], y[i]

    return X, y


def train():
    timer.start()
    print('started preparing data')
    X, y = prepare_data('features_full/train/face/data.npy', 'features_full/train/non-face/data.npy')
    print('done preparing data', str(timer.stop()) + str('s'))

    ada_boost = AdaBoostClassifier()
    timer.start()
    print('started training')
    ada_boost.fit(X, y)
    print('done training', str(timer.stop()) + str('s'))

    with open('ada_boosts/params1.pickle', 'wb') as handle:
        pickle.dump(ada_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_score(obj_file_name):
    with open('ada_boosts/' + obj_file_name, 'rb') as handle:
        ada_boost = pickle.load(handle)

    X, y = prepare_data('features_full/test/face/data.npy', 'features_full/test/non-face/data.npy')

    print('score:', ada_boost.score(X, y))


# train()
get_score('params1.pickle')
