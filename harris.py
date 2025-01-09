import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"程序执行时间: {self.execution_time}秒")

def sobel_filters():
    """ sobel算子计算水平、垂直方向梯度 """
    sobel_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    return sobel_x, sobel_y

def compute_gradient(image):

    sobel_x, sobel_y = sobel_filters()

    # 卷积
    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)
    cv2.imwrite("harris_horizontal_grads.jpg", grad_x)
    cv2.imshow("sobel_x", grad_x)
    cv2.imwrite("harris_vertical_grads.jpg", grad_y)
    cv2.imshow('sobel_y', grad_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return grad_x, grad_y

def compute_harris_response(grad_x, grad_y, k=0.04):
    """ 计算每个pixel的harris响应 """
    Ixx = grad_x ** 2
    Ixy = grad_x * grad_y
    Iyy = grad_y ** 2

    # 用高斯函数求和平滑噪声
    gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=1)

    Sxx = cv2.filter2D(Ixx, -1, gaussian_kernel)
    Sxy = cv2.filter2D(Ixy, -1, gaussian_kernel)
    Syy = cv2.filter2D(Iyy, -1, gaussian_kernel)
    # 计算响应
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    print(Sxx.shape, Syy.shape, Sxy.shape)
    print(R.shape)
    cv2.imshow('Sxx', Sxx)
    cv2.imshow('Syy', Syy)
    cv2.imshow('Sxy', Sxy)
    cv2.imwrite('harris_response.jpg', R)
    cv2.imshow('harris_response', R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return R

def non_maximum_suppression(response, neighborhood_size=5):
    """ NMS优化结果 """
    R_nms = np.zeros(response.shape, dtype=np.float32)
    half_size = neighborhood_size // 2

    for y in range(half_size, response.shape[0] - half_size):
        for x in range(half_size, response.shape[1] - half_size):
            local_region = response[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]
            if response[y, x] == np.max(local_region):
                R_nms[y, x] = response[y, x]
                
    # 获取前250和前500响应位置
    indices_250 = np.unravel_index(np.argsort(R_nms, axis=None)[-250:], R_nms.shape)
    indices_500 = np.unravel_index(np.argsort(R_nms, axis=None)[-500:], R_nms.shape)

    strongest_250 = list(zip(indices_250[0], indices_250[1]))
    strongest_500 = list(zip(indices_500[0], indices_500[1]))
    
    return R_nms, strongest_250, strongest_500

def harris_corners(image, threshold=0.01):
    """ 角点检测 """
    corners_nms = []
    strongest_250_list = []
    strongest_500_list = []
    supp_rads = []
    grad_x, grad_y = compute_gradient(image)
    response = compute_harris_response(grad_x, grad_y)
    # 保留大于阈值的数
    corners = np.where(response > threshold * np.max(response), response, 0)
    # NMS
    for supp_rad in range(5, 25, 5):
        supp_rads.append(supp_rad)
        corners_nms_i, strongest_250, strongest_500 = non_maximum_suppression(corners, neighborhood_size=supp_rad)
        corners_nms.append(corners_nms_i)
        strongest_250_list.append(strongest_250)
        strongest_500_list.append(strongest_500)
    return corners, corners_nms, strongest_250_list, strongest_500_list, supp_rads

def draw_corners(image, corners):
    """ 画点 """
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for y in range(corners.shape[0]):
        for x in range(corners.shape[1]):
            if corners[y, x] != 0:
                cv2.circle(result, (x, y), 3, (255, 0, 0), 1)
    return result

def draw_strongest_corners(image, strongest_corners):
    """ 画出最强角点 """
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for y, x in strongest_corners:
        cv2.circle(result, (x, y), 3, (255, 0, 0), 1)
    return result

if __name__ == '__main__':
    with Timer():
        # 读取并处理图像
        image = cv2.imread(
            "ac1329024dce07bc6cbb31f1796e59bb_r.jpg", cv2.IMREAD_GRAYSCALE
        )  # 将图像读取为灰度图
        image = cv2.resize(image,(512,512))
        cv2.imwrite('resized_image.jpg', image)
        # 绘制角点
        corners, corners_nms, strongest_250_list, strongest_500_list, supp_rads = harris_corners(image, threshold=0.01)
        image_with_corners = draw_corners(image, corners)
        # 显示结果
        plt.imshow(image_with_corners)
        plt.title("Harris Corners")
        cv2.imwrite('harris_corners.jpg', image_with_corners)
        plt.axis("off")
        plt.show()
        for corners_nms_i, strongest_250, strongest_500, supp_rad in zip(corners_nms, strongest_250_list, strongest_500_list, supp_rads):
            image_with_corners_nms = draw_corners(image, corners_nms_i)
            plt.imshow(image_with_corners_nms)
            plt.title(f'Harris Corners with NMS in {supp_rad} supp_rad')
            cv2.imwrite(f"Harris Corners with NMS in {supp_rad} supp_rad.jpg", image_with_corners_nms)
            plt.axis('off')
            plt.show()
            image_with_strongest_250 = draw_strongest_corners(image, strongest_250)
            plt.imshow(image_with_strongest_250)
            plt.title(f"Strongest 250 Corners with NMS in {supp_rad} supp_rad")
            cv2.imwrite(f"Strongest 250 Corners with NMS in {supp_rad} supp_rad.jpg", image_with_strongest_250)
            plt.axis('off')
            plt.show()
            image_with_strongest_500 = draw_strongest_corners(image, strongest_500)
            plt.imshow(image_with_strongest_500)
            plt.title(f"Strongest 500 Corners with NMS in {supp_rad} supp_rad")
            cv2.imwrite(f"Strongest 500 Corners with NMS in {supp_rad} supp_rad.jpg", image_with_strongest_500)
            plt.axis('off')
            plt.show()
