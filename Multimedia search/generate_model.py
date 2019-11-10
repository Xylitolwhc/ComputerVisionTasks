import cv2
import numpy as np
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import time
import pathlib
import hierarchical_cluster as hc

TRAIN_PATH = "dataset/training/"
HIERARCHY_NODES = 3
HIERARCHY_DEPTH = 6


# 提取图片sift特征
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
    descriptors = None

    # 提取图片sift特征
    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print("Extract SIFT of %s image, %d of %d images" % (image_names[i], i + 1, len(image_paths)), end=".\t")
        kp, des = des_ext.detectAndCompute(im, None)
        print("Extracted %d features" % len(des))
        des_list.append((image_path, des))
        if descriptors is None:
            descriptors = np.array(des)
        else:
            descriptors = np.vstack((descriptors, des))

    # # 将存有所有sift特征的list转换为numpy数组
    # print("Transform list into nparray...")
    # descriptors = np.array(descriptors)

    # 将提取出的特征保存至文件descriptors.xz
    joblib.dump((image_paths, des_list, descriptors), "descriptors.joblib", compress=False)
    return image_paths, des_list, descriptors


# 进行层次聚类
def perform_hierarchical_cluster(image_paths, des_list, descriptors):
    # 开始层次聚类
    print("Start hierarchical clustering: %d key points" % descriptors.shape[0])
    centroids = hc.hierarchical_cluster(descriptors, descriptors_num=None, nodes=HIERARCHY_NODES,
                                        max_level=HIERARCHY_DEPTH)

    # BoW 编码
    print("Start BoW encoding...")
    im_features = hc.hierarchical_cluster_transform(centroids, des_list)

    # Tf-Idf 变换 & L2 归一化
    print("Start Tf-idf transforming...")
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    joblib.dump((im_features, image_paths, idf, centroids), "model.joblib", compress=3)
    return im_features, image_paths, idf, centroids


def main():
    start_time = time.time()

    # image_paths, des_list, descriptors = perform_sift()
    image_paths, des_list, descriptors = joblib.load("descriptors.joblib")
    im_features, image_paths, idf, centroids = \
        perform_hierarchical_cluster(image_paths, des_list, descriptors)

    end_time = time.time()
    print("Time used:", end_time - start_time)


if __name__ == '__main__':
    main()
