import cv2
import numpy as np
import os
import joblib
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import *
from sklearn import preprocessing
import time
import pathlib

TRAIN_PATH = "dataset/training/"
numWords = 200


def perform_sift():
    # 读取TRAIN_PATH下所有训练集图片路径，保存于数组中
    data_root = pathlib.Path(TRAIN_PATH)
    image_paths = list(data_root.glob('*'))
    image_paths = [str(path) for path in image_paths]
    image_names = [pathlib.Path(path).stem for path in image_paths]

    # 创建用于提取sift特征的对象
    des_ext = cv2.xfeatures2d.SIFT_create()

    # 创建用于保存所有sift特征的list
    des_list = []
    descriptors = []

    # 提取图片sift特征
    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print("Extract SIFT of %s image, %d of %d images" % (image_names[i], i + 1, len(image_paths)))
        kp, des = des_ext.detectAndCompute(im, None)
        des_list.append((image_path, des))
        descriptors.append(des)

    # 将存有所有sift特征的list转换为numpy数组
    descriptors = np.array(descriptors)

    # 将提取出的特征保存至文件descriptors.xz
    joblib.dump((image_paths, des_list, descriptors), "descriptors.joblib", compress=False)
    return image_paths, des_list, descriptors


def perform_hierarchical_cluster(image_paths, des_list, descriptors):
    # 开始层次聚类
    print("Start hierarchical clustering: %d words, %d key points" % (numWords, descriptors.shape[0]))
    centroids = hierarchical_cluster(descriptors)

    im_features = np.zeros((len(image_paths), numWords), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], centroids)
        for w in words:
            im_features[i][w] += 1

    pass


# 虚假的层次聚类
def hierarchical_cluster(descriptors, nodes, max_level, current_level=0):
    level_info = {}
    level_info["level"] = current_level

    centroid, label = kmeans2(descriptors, nodes, 1)
    level_info["centroids"] = centroid

    classified_descriptors = {}
    for i in range(nodes):
        classified_descriptors[i] = []
    for i in len(label):
        classified_descriptors[label[i]].append(descriptors[i])

    if current_level != max_level:
        for i in range(nodes):
            level_info[i] = hierarchical_cluster(classified_descriptors[i], nodes, current_level + 1)
    return level_info


# 真正的层次聚类
def hierarchical_cluster(descriptors, max_cluster):
    labels = fclusterdata(descriptors, t=max_cluster, criterion="maxclust")

    # 将sift特征按照聚类结果分类存放
    classfied_des_num = {}
    for i in range(len(labels)):
        label = labels[i]
        if label in classfied_des_num:
            classfied_des_num[label].append(i)
        else:
            classfied_des_num[label] = [i]

    centroids = []
    # 计算每一类的类中心
    for label in classfied_des_num.keys():
        indexes = classfied_des_num[label]
        sum = descriptors[indexes[0]]
        for i in indexes[1:]:
            sum += descriptors[indexes[i]]
        sum = sum / len(indexes)
        centroids.append(sum)
    centroids = np.array(centroids)

    return centroids


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
    # perform_hierarchical_cluster(image_paths, des_list, descriptors)
    # perform_kmeans(image_paths, des_list, descriptors)
    end_time = time.time()
    print("Time used:", end_time - start_time)


if __name__ == '__main__':
    main()
