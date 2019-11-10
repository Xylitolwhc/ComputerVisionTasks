import joblib
import cv2
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import hierarchical_cluster as hc
import pathlib
import re
import time

TEST_PATH = "dataset/testing"

RERANK_DEPTH = 9
MIN_MATCH_COUNT = 10


def search(image_path):
    # 读取训练好的分类模型
    im_features, image_paths, idf, centroids = joblib.load("model.joblib")

    # 提取需要查询的图片的sift特征
    des_ext = cv2.xfeatures2d.SIFT_create()
    image = cv2.imread(image_path)
    kp, des = des_ext.detectAndCompute(image, None)

    # 使用BoW进行编码
    features = hc.hierarchical_cluster_transform(centroids, [[0, des]])
    features = features * idf
    features = preprocessing.normalize(features, norm='l2')

    # 查找最相近的图片
    score = np.dot(features, im_features.T)
    rank_ID = np.argsort(-score)

    # 使用RANSAC空间校验对部分匹配到的图片进行二次排序
    reRank_image_path = []
    reRank_image_score = np.zeros([RERANK_DEPTH])
    matcher = cv2.BFMatcher()
    for i, ID in enumerate(rank_ID[0][0:RERANK_DEPTH]):
        reRank_image_path.append(image_paths[ID])
        img = plt.imread(image_paths[ID])
        kp1, des1 = des_ext.detectAndCompute(img, None)
        # 暴力搜索，找到所有最近的匹配点
        knn_matches = matcher.knnMatch(des, des1, 2)
        ratio_thresh = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCH_COUNT:
            # 使用RANSAC对所有匹配点进行校验，筛选出有效的匹配点对
            src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            # 统计有效匹配点对数量
            inliers_cnt = np.sum(matchesMask)
            reRank_image_score[i] = inliers_cnt / len(matchesMask)
        else:
            reRank_image_score[i] = 0

    reRank_ID = np.argsort(-reRank_image_score)
    return reRank_ID, reRank_image_path, reRank_image_score, image


def show_search_result(reRank_ID, reRank_image_path, reRank_image_score, image):
    plt.figure(num=1, figsize=(10, 20))
    plt.gray()
    plt.subplot(4, 3, 2)
    plt.imshow(image[:, :, ::-1])
    plt.axis('off')
    # 显示搜索结果
    for i, ID in enumerate(reRank_ID):
        img_name = pathlib.Path(reRank_image_path[ID]).stem
        img = plt.imread(reRank_image_path[ID])
        plt.gray()
        plt.subplot(4, 3, i + 4)
        plt.imshow(img)
        plt.axis('off')
        plt.title(img_name + " score:%.3f" % reRank_image_score[ID])

    # plt.savefig("./a.jpg")
    plt.show()


def show_RANSAC(img_path1, img_path2):
    img_object = cv2.imread(img_path1)
    img_scene = cv2.imread(img_path2)
    # -- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    detector = cv2.xfeatures2d.SIFT_create()
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SIFT is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    print(len(good_matches))
    # -- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1] + img_scene.shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
    # -- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = img_object.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = img_object.shape[1]
    obj_corners[2, 0, 1] = img_object.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = img_object.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, H)
    # -- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv2.line(img_matches, (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])), \
             (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])), \
             (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])), \
             (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
    cv2.line(img_matches, (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])), \
             (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

    # -- Show detected matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey()


def test_all():
    correct_cnt = 0
    test_image_paths = list(pathlib.Path(TEST_PATH).glob("*"))
    for image_path in test_image_paths:
        reRank_ID, reRank_image_path, reRank_image_score = search(str(image_path))
        query_img_name = pathlib.Path(image_path).stem
        found_img_name = pathlib.Path(reRank_image_path[reRank_ID[1]]).stem

        query_label = query_img_name[:re.search("_\d", query_img_name).start()]
        found_label = found_img_name[:re.search("_\d", found_img_name).start()]

        if query_label == found_label:
            correct_cnt += 1
        print(query_label == found_label)
    print(correct_cnt)
    print(len(test_image_paths))


def test():
    test_image_path = "dataset/testing/radcliffe_camera_000397.jpg"
    reRank_ID, reRank_image_path, reRank_image_score, image = search(test_image_path)
    show_search_result(reRank_ID, reRank_image_path, reRank_image_score, image)


def main():
    # test_all()
    test()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Time used:", end_time - start_time)
