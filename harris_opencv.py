import cv2
import numpy as np
import matplotlib.pyplot as plt
from harris import Timer

with Timer():
    # 读取图像并转换为灰度图
    # image = cv2.imread("ac1329024dce07bc6cbb31f1796e59bb_r.jpg")
    image = cv2.imread(r"h:\c++\cuda_harris\input\image1.ppm")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 转换图像数据类型为 float32
    gray_image = np.float32(gray_image)

    # 设置 Harris 角点检测的参数
    block_size = 2  # 计算局部微分矩阵的领域大小
    ksize = 3  # Sobel 卷积核的大小
    k = 0.04  # Harris 角点检测中的参数

    # 执行 Harris 角点检测
    harris_corners = cv2.cornerHarris(gray_image, block_size, ksize, k)

    # 进行扩展，以便查看所有检测到的角点
    harris_corners = cv2.dilate(harris_corners, None)

    # 设置阈值以显示角点
    threshold = 0.01 * harris_corners.max()  # 0.01 乘以最大值作为阈值
    image_with_corners = image.copy()
    image_with_corners[harris_corners > threshold] = [0, 0, 255]  # 用红色标记角点

    # 显示结果
    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corners")
    # cv2.imwrite('harris_result_opencv.jpg', image_with_corners)
    plt.axis("off")
    plt.show()
