from scipy.cluster.vq import *
import numpy as np


# 虚假的层次聚类，其实应该写成个class
def hierarchical_cluster(descriptors, descriptors_num, nodes, max_level, current_level=0):
    level_info = {}
    level_info["level"] = current_level
    level_info["nodes"] = nodes

    # 如果数据小于节点数则将当前节点作为根节点
    if len(descriptors) > nodes and current_level < max_level:
        level_info["isEnd"] = False
    else:
        level_info["isEnd"] = True
        return level_info

    # 根据序号取出需要聚类的数据
    if descriptors_num is None:
        descriptors_num = [i for i in range(len(descriptors))]
        data = descriptors
    else:
        data = [descriptors[i] for i in descriptors_num]

    # k-means聚类
    # print("K-means at level %d" % current_level)
    centroids, labels = kmeans2(data, nodes, 3)
    level_info["centroids"] = centroids

    # 根据聚类结果将数据分类存放，仅存储序号
    classified_descriptors_num = {}
    for i, label in enumerate(labels):
        if label in classified_descriptors_num.keys():
            classified_descriptors_num[label].append(descriptors_num[i])
        else:
            classified_descriptors_num[label] = [descriptors_num[i]]

    # 进行下一层聚类
    for i in range(nodes):
        level_info[i] = hierarchical_cluster(descriptors, classified_descriptors_num[i], nodes, max_level,
                                             current_level + 1)

    if current_level == 0:
        labels = hierarchical_cluster_number(level_info)
        level_info["words"] = labels
        print("Hierarchical cluster finished, %d words generated." % labels)
    return level_info


# 对层次聚类结果进行编号
def hierarchical_cluster_number(level_info, cluster_num=0):
    nodes = level_info["nodes"]
    if level_info["isEnd"] is False:
        for i in range(nodes):
            cluster_num = hierarchical_cluster_number(level_info[i], cluster_num)
    else:
        level_info["num"] = cluster_num
        cluster_num += 1
    return cluster_num


# 虚假的层次聚类预测
def hierarchical_cluster_predict(level_info, descriptors):
    labels = []
    for descriptor in descriptors:
        node = level_info
        while node["isEnd"] is False:
            centroids = node["centroids"]
            label, dist = vq([descriptor], centroids)
            node = node[label[0]]
        labels.append(node["num"])
    return labels


# BoW 编码
def hierarchical_cluster_transform(centroids, des_list):
    im_features = np.zeros((len(des_list), centroids["words"]), "float32")
    for i in range(len(des_list)):
        words = hierarchical_cluster_predict(centroids, des_list[i][1])
        for w in words:
            im_features[i][w] += 1
    return im_features

# 真正的层次聚类
# def real_hierarchical_cluster(descriptors, max_cluster):
#     labels = fclusterdata(descriptors, t=max_cluster, criterion="maxclust")
#
#     # 将sift特征按照聚类结果分类存放
#     classfied_des_num = {}
#     for i in range(len(labels)):
#         label = labels[i]
#         if label in classfied_des_num:
#             classfied_des_num[label].append(i)
#         else:
#             classfied_des_num[label] = [i]
#
#     centroids = []
#     # 计算每一类的类中心
#     for label in classfied_des_num.keys():
#         indexes = classfied_des_num[label]
#         sum = descriptors[indexes[0]]
#         for i in indexes[1:]:
#             sum += descriptors[indexes[i]]
#         sum = sum / len(indexes)
#         centroids.append(sum)
#     centroids = np.array(centroids)
#
#     return centroids
