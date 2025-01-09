#!/usr/bin/env python3
# coding: utf-8

import cv2
import copy
import numpy as np
import time

def bgr2gray(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    gray = gray.astype(np.uint8)
    return gray


def sobel_filtering(gray):
    # get shape
    img_h, img_w = gray.shape

    # sobel kernel
    sobel_y = np.array(((1, 2, 1), (0, 0, 0), (-1, -2, -1)), dtype=np.float32)

    sobel_x = np.array(((1, 0, -1), (2, 0, -2), (1, 0, -1)), dtype=np.float32)

    # padding
    tmp = np.pad(gray, (1, 1), "edge")

    # prepare
    ix = np.zeros_like(gray, dtype=np.float32)
    iy = np.zeros_like(gray, dtype=np.float32)

    # get differential
    for y in range(img_h):
        for x in range(img_w):
            ix[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobel_x)
            iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobel_y)

    ix2 = ix**2
    iy2 = iy**2
    ixy = ix * iy
    return ix2, iy2, ixy


def gaussian_filtering(I, k_size=3, sigma=3):
    # get shape
    img_h, img_w = I.shape

    # gaussian
    i_t = np.pad(I, (k_size // 2, k_size // 2), "edge")

    # gaussian kernel
    K = np.zeros((k_size, k_size), dtype=np.float32)
    for x in range(k_size):
        for y in range(k_size):
            _x = x - k_size // 2
            _y = y - k_size // 2
            K[y, x] = np.exp(-(_x**2 + _y**2) / (2 * (sigma**2)))
    K /= sigma * np.sqrt(2 * np.pi)
    K /= K.sum()

    # filtering
    for y in range(img_h):
        for x in range(img_w):
            I[y, x] = np.sum(i_t[y : y + k_size, x : x + k_size] * K)
    return I


def corner_detect(img, ix2, iy2, ixy, k=0.04, th=0.1):
    # prepare output image
    out = copy.deepcopy(img)

    # get R
    R = (ix2 * iy2 - ixy**2) - k * ((ix2 + iy2) ** 2)

    # detect corner
    out[R >= np.max(R) * th] = [255, 0, 0]
    out = out.astype(np.uint8)
    return out


def harris_corner(img):
    # 1. grayscale
    gray = bgr2gray(img)

    # 2. get difference image
    ix2, iy2, ixy = sobel_filtering(gray)

    # 3. gaussian filtering
    ix2 = gaussian_filtering(ix2, k_size=3, sigma=3)
    iy2 = gaussian_filtering(iy2, k_size=3, sigma=3)
    ixy = gaussian_filtering(ixy, k_size=3, sigma=3)

    # 4. corner detect
    out = corner_detect(img, ix2, iy2, ixy)
    return out


def main():
    # Read image
    img = cv2.imread(r"h:\c++\cuda_harris\input\image1.ppm")
    # img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)

    # Harris corner detection
    out = harris_corner(img)
    cv2.imwrite("out.jpg", out)
    # cv2.imshow("out.jpg", out)
    print("proc ok.")
    # cv2.waitKey(0)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time}秒")
