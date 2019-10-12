import numpy as np
import cv2
from matplotlib import pyplot as plt

def sift(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img2 = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))
    # img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Sift", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fast0(img):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
    cv2.imshow("Sift", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('./img/tree/img1.ppm',0)
fast0(img)
#sift()
