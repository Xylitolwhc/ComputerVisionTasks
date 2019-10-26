# Source:https://github.com/willard-yuan/image-retrieval/tree/master/bag-of-words-python-dev-version
# python 3.7.4
# opencv-contrib 4.1.2
# opencv 4.1.1

import cv2
import numpy as np
import os
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import time

train_path = "dataset/logo_dataset/"
training_names = os.listdir(train_path)
numWords = 200


def perform_sift():
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    for training_name in training_names:
        image_path = os.path.join(train_path, training_name)
        image_paths += [image_path]

    # Create feature extraction and keypoint detector objects
    des_ext = cv2.xfeatures2d.SIFT_create()

    # List where all the descriptors are stored
    des_list = []

    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print("Extract SIFT of %s image, %d of %d images" % (training_names[i], i + 1, len(image_paths)))
        kp, des = des_ext.detectAndCompute(im, None)
        des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    joblib.dump((image_paths, des_list, descriptors), "descriptors.xz")
    return image_paths, des_list, descriptors


def perform_kmeans(image_paths, des_list, descriptors):
    # Perform k-means clustering
    print("Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
    voc, variance = kmeans(descriptors, numWords, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), numWords), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    joblib.dump((im_features, image_paths, idf, numWords, voc), "200words.xz", compress=3)


def main():
    start_time = time.time()
    image_paths, des_list, descriptors = perform_sift()
    # image_paths, des_list, descriptors = joblib.load("descriptors.xz")
    perform_kmeans(image_paths, des_list, descriptors)
    end_time = time.time()
    print("Time used:", end_time - start_time)


if __name__ == '__main__':
    main()
