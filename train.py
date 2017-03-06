import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle

from Timer import timer


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


def train(file_name):
    timer.start()
    print('started preparing data')
    X, y, w = prepare_data('features_full/train/face/data.npy', 'features_full/train/non-face/data.npy')
    print('done preparing data', str(timer.stop()) + str('s'))

    ada_boost = AdaBoostClassifier()
    timer.start()
    print('started training')
    ada_boost.fit(X, y, w)
    print('done training', str(timer.stop()) + str('s'))

    with open('ada_boosts/' + file_name + '.pickle', 'wb') as handle:
        pickle.dump(ada_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)


train('')
