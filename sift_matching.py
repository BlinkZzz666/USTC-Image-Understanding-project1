import numpy as np
import cv2
import sift
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('matching_demo1.jpg', 0)
img2 = cv2.imread('matching_demo2.jpg', 0)

keypoint1, descriptor1 = sift.computeKeypointsAndDescriptors(img1)
keypoint2, descriptor2 = sift.computeKeypointsAndDescriptors(img2)

# 采用 FLANN 匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #定义搜索树的大小
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params) 
matches = flann.knnMatch(descriptor1, descriptor2, k=2)  # 使用 k-NN匹配， 每个特征点匹配两个对应点

# lowe测试下来最好的比例系数
good = []
for m, n in matches: # 最佳匹配点的distance比次佳匹配点的0.7大就说明匹配成功，因为两个匹配点之间的距离也很近
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # 通过两对特征点来计算homography矩阵
    src_pts = np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # 可视化对应特征点
    h, w = img1.shape
    pts = np.float32([[0, 0], 
                      [0, h - 1], 
                      [w - 1, h - 1], 
                      [w - 1, 0]]).reshape(-1, 1, 2)
    # 根据 Homography 对特征点应用透视变换
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    # 拼接图像
    for i in range(3):
        newimg[hdif : hdif + h1, :w1, i] = img1
        newimg[:h2, w1 : w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    # 这里 +w1, hdif 都是因为两张图像拼接是引起的偏差
    # 画最好的50个匹配点(distance最小)
    num_matches = 50
    good_matches = sorted(good, key=lambda x: x.distance)[:num_matches]
    for m in good_matches:

        pt1 = (int(keypoint1[m.queryIdx].pt[0]), int(keypoint1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(keypoint2[m.trainIdx].pt[0] + w1), int(keypoint2[m.trainIdx].pt[1]))

        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    print('good matches: %d, plotted matches: %d' % (len(good), len(good_matches)))
    plt.imshow(newimg)
    plt.savefig('matching_result.jpg')
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
