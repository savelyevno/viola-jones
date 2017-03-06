from PIL import Image
import numpy as np
import os
from Timer import ticker, timer


project_path = '/home/nikita/PycharmProjects/viola-jones/'
input_folder = 'faces/'
output_folder_full = 'features_full/'
output_folder_partial = 'features_partial/'


def get_feature_vector(img):
    def get_ii(img):
        ii = np.empty((img.shape[0], img.shape[1]), dtype=np.int32)

        ii[0, 0] = img[0, 0]
        for i in range(1, img.shape[0]):
            ii[i, 0] = ii[i - 1, 0] + img[i, 0]
        for j in range(1, img.shape[1]):
            ii[0, j] = ii[0, j - 1] + img[0, j]
        for i in range(1, img.shape[0]):
            for j in range(1, img.shape[1]):
                ii[i, j] = img[i, j] + ii[i - 1, j] + ii[i, j - 1] - ii[i - 1, j - 1]
                
        return ii

    ii = get_ii(img)

    def get_rect_sum(i1, j1, i2, j2):
        # inclusive
        if i1 == 0:
            if j1 == 0:
                return ii[i2, j2]
            else:
                return ii[i2, j2] - ii[i2, j1 - 1]
        else:
            if j1 == 0:
                return ii[i2, j2] - ii[i1 - 1, j2]
            else:
                return ii[i2, j2] - ii[i1 - 1, j2] - ii[i2, j1 - 1] + ii[i1 - 1, j1 - 1]
    
    result = np.empty(63960, dtype=np.int16)
    cnt = 0
    for h in range(1, img.shape[1] + 1):
        for w in range(1, img.shape[0] + 1):
            #
            #   A
            #
            for i in range(0, img.shape[0] - h + 1):
                for j in range(0, img.shape[1] - 2 * w + 1):
                    result[cnt] = get_rect_sum(i, j, i + h - 1, j + w - 1) - \
                                                get_rect_sum(i, j + w, i + h - 1, j + 2 * w - 1)
                    cnt += 1

            #
            #   B
            #
            for i in range(0, img.shape[0] - 2 * h + 1):
                for j in range(0, img.shape[1] - w + 1):
                    result[cnt] = get_rect_sum(i + h, j, i + 2 * h - 1, j + w - 1) - \
                                                get_rect_sum(i, j, i + h - 1, j + w - 1)
                    cnt += 1

            #
            #   C
            #
            for i in range(0, img.shape[0] - h + 1):
                for j in range(0, img.shape[1] - 3 * w + 1):
                    result[cnt] = get_rect_sum(i, j, i + h - 1, j + w - 1) - \
                                                get_rect_sum(i, j + w, i + h - 1, j + 2 * w - 1) + \
                                                get_rect_sum(i, j + 2 * w, i + h - 1, j + 3 * w - 1)
                    cnt += 1

            #
            #   C'
            #
            for i in range(0, img.shape[0] - 3 * h + 1):
                for j in range(0, img.shape[1] - w + 1):
                    result[cnt] = get_rect_sum(i, j, i + h - 1, j + w - 1) - \
                                                get_rect_sum(i + h, j, i + 2 * h - 1, j + w - 1) + \
                                                get_rect_sum(i + 2 * h, j, i + 3 * h - 1, j + w - 1)
                    cnt += 1

            #
            #   D
            #
            for i in range(0, img.shape[0] - 2 * h + 1):
                for j in range(0, img.shape[1] - 2 * w + 1):
                    result[cnt] = get_rect_sum(i, j, i + h - 1, j + w - 1) - \
                                                get_rect_sum(i, j + 2, i + h - 1, j + 2 * w - 1) + \
                                                get_rect_sum(i + h, j, i + 2 * h - 1, j + w - 1) - \
                                                get_rect_sum(i + h, j + w, i + 2 * h - 1, j + 2 * w - 1)
                    cnt += 1

    return result


def main():
    def process_folder(rel_path):
        tick = ticker.start_track(10, lambda: print('\t' + str(file_count)))
        timer.start()

        print(rel_path, (len(os.listdir(input_folder + rel_path))))

        features = np.empty((len(os.listdir(input_folder + rel_path)), 63960), dtype=np.int16)
        file_count = -1
        for filename in os.listdir(input_folder + rel_path):
            file_count += 1
            img = np.uint8(Image.open(input_folder + rel_path + '/' + filename))

            feature_vector = get_feature_vector(img)
            np.save(output_folder_partial + rel_path + '/' + filename.split('.')[0], feature_vector)

            features[file_count] = feature_vector

        np.save(output_folder_full + rel_path + '/' + 'data', features)

        ticker.stop_track(tick)
        print(timer.stop())

    for subdir, dirs, files in os.walk(input_folder):
        if 100 < len(files):
            process_folder(os.path.relpath(subdir, input_folder))


main()
