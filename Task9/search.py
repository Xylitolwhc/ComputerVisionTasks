#!/usr/local/bin/python2.7
# python search.py -i dataset/train/ukbench00000.jpg

import joblib
import cv2
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Get query image path
image_path = "dataset/logo_dataset/310100_07_01_01_16777220_40942_102023-2 (Small).jpg"

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

# Create feature extraction and keypoint detector objects
des_ext = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
kp, des = des_ext.detectAndCompute(im, None)

des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

# 
test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features * idf
test_features = preprocessing.normalize(test_features, norm='l2')

score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)

# Visualize the results
plt.figure(figsize=(10, 20))
plt.gray()
plt.subplot(4, 3, 1)
plt.imshow(im[:, :, ::-1])
plt.axis('off')
for i, ID in enumerate(rank_ID[0][0:9]):
    img = plt.imread(image_paths[ID])
    plt.gray()
    plt.subplot(4, 3, i + 4)
    plt.imshow(img)
    plt.axis('off')

plt.savefig("./a.jpg")
plt.show()
