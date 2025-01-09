import cv2
import matplotlib.pyplot as plt
from harris import Timer

if __name__ == '__main__':
    with Timer():
        # 读取图像
        image = cv2.imread("ac1329024dce07bc6cbb31f1796e59bb_r.jpg", cv2.IMREAD_GRAYSCALE)

        # 创建 SIFT 检测器
        sift = cv2.SIFT_create()

        # 检测关键点和计算描述符
        keypoints, descriptors = sift.detectAndCompute(image, None)

        output_image = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # 显示结果
        plt.imshow(output_image, cmap="gray")
        plt.title("SIFT Keypoints")
        plt.axis("off")
        plt.show()

