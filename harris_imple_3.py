import numpy as np
import cv2
import matplotlib.pyplot as plt

def harris_corner_detect(img, block_size=3, ksize=3, k=0.04, threshold=0.01, with_NMS=False):
    '''
    params:
        img:单通道灰度图片
        block_size:权重滑动窗口
        ksize：Sobel算子窗口大小
        k:响应函数参数k
        threshold:设定阈值
        WITH_NMS:非极大值抑制
    return：
        corner：角点位置图，与源图像一样大小，角点处像素值设置为255
    '''
    h,w = img.shape[:2]
    # 1.高斯平滑
    gray = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=2)

    # 2.计算梯度
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize) # x方向梯度
    # 将梯度值转换为可显示的格式
    grad_x = cv2.convertScaleAbs(grad[:, :, 0])  # 转换为绝对值并转换为8位图像

    # 显示x方向的梯度图像
    cv2.imshow("Gradient in X Direction", grad_x)

    # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    grad[:, :, 1] = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize) # y方向梯度
    grad_y = cv2.convertScaleAbs(grad[:, :, 1])  # 转换为绝对值并转换为8位图像
    cv2.imshow("Gradient in Y Direction", grad_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 3.计算hessian矩阵
    hessian = np.zeros((h, w, 3), dtype=np.float32)
    hessian[:, :, 0] = grad[:, :, 0] ** 2 # Sxx
    hessian[:, :, 1] = grad[:, :, 1] ** 2 # Syy
    hessian[:, :, 2] = grad[:, :, 0] * grad[:, :, 1] # Sxy
    
    hessian = [np.array([[hessian[i, j, 0], hessian[i, j, 2]], [hessian[i, j ,2], hessian[i, j, 1]]]) for i in range(h) for j in range(w)]

    print(hessian[1].shape)

    # 计算 hessian矩阵的行列式和迹，以及harris响应R
    det, trace = list(map(np.linalg.det, hessian)), list(map(np.trace, hessian))
    R = np.array([d - k * t ** 2 for d, t in zip(det, trace)])
    R = abs(R)
    # 5.将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    # NMS
    for i in range(h):
        for j in range(w):
            if with_NMS:
                # 除了进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i, j] > R_max * threshold and R[i, j] == np.max(R[max(0, i-1): min(i+2, h-1), max(0, j-1): min(j+2, w-1)]):
                    corner[i, j] = 255
            else:
                # 只进行阈值检测
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255

    return corner

if __name__ == '__main__':
    image = cv2.imread("ac1329024dce07bc6cbb31f1796e59bb_r.jpg")
    h, w, c = image.shape
    print("image shape --> h:%d  w:%d  c:%d" % (h, w, c))
    cv2.imshow("original_image", image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    dst = harris_corner_detect(gray)
    
    image_dst = image[:, :, :]
    image_dst[dst > 0.01 * dst.max()] = [0, 0, 255]
    # cv2.imwrite('./dsti8_1.jpg', image_dst)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
