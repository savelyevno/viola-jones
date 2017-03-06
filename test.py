import numpy as np
import pickle
import os
import sys

from Timer import timer, ticker


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


test_model(sys.argv[1] + '.pickle')
