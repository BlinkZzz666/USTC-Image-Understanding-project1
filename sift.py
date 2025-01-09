import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import cmp_to_key
from harris import Timer

float_tolerance = 1e-7


# Image pyramid
def generate_base_image(image, sigma, assumed_blur):  # assumed_blur是输入图像的初始模糊
    """upsample and blur"""
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(
        max((sigma**2) - ((2 * assumed_blur) ** 2), 0.01)
    )  # 需要的模糊程度是sigma ** 2，若 sigma ** 2 = sigma_1 ** 2 + sigma_2 **2,可以证明等价于先按sigma_1 ** 2 模糊一次，再按sigma_2 ** 2 模糊一次

    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def compute_num_of_octaves(image_shape):
    """num of octaves in image pyramid ,计算公式log2 [min(w, h)]"""
    return int(
        round(np.log(min(image_shape)) / np.log(2) - 1)
    )  # 最顶层图像至少要有1个pixel，所以 y / (2 ** x) = 1，根据此求出x。减1是因为要将x向下取整，保证金字塔有整数层


def generate_Gaussian_kernels(sigma, num_intervals):
    num_image_per_octave = (
        num_intervals + 3
    )  # 假设 高斯金字塔有n层图片，由于DoG金字塔是前一张高斯图片 - 后一张高斯图片，所以DoG金字塔有n-1层，由于每3张DoG图片（前、中、后）决定一个interval（interval用于检测极值点），所以第一张和最后一张没办法检测，因此最终是n-3个intervals，若要n个interval，就要n+3张高斯图片
    k = 2 ** (1.0 / num_intervals)
    gaussian_kernels = np.zeros(num_image_per_octave)  # initialize gaussian kernel
    gaussian_kernels[0] = sigma  # first
    for image_index in range(1, num_image_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total**2 - sigma_previous**2)
    return gaussian_kernels


def generate_Gaussian_images(image, num_octaves, gaussian_kernels):
    """scale space pyramid of gaussian images"""

    gaussian_images = []

    for octave_index in range(
        num_octaves
    ):  # 这里将基础图像缩小了num_octaves - 1倍，所以最顶层的图像至少是3x3像素的，符合之后的计算策略
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(
            image
        )  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(
                image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel
            )
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[
            -3
        ]  # 下一octave的第0层是上一octave的倒数第三层downsample得到
        image = cv2.resize(
            octave_base,
            (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.array(gaussian_images, dtype=object)


def generate_DoG_images(gaussian_images):
    """Generate DoG image pyramid"""
    dog_images = []  # 如高斯金字塔的注释所说，DoG金字塔每个octave比高斯金字塔少一张图片
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(
            gaussian_images_in_octave, gaussian_images_in_octave[1:]
        ):
            dog_images_in_octave.append(
                cv2.subtract(second_image, first_image)
            )  # 这里数据类型是float32，因此用opencv的减法保证数据不会溢出
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images, dtype=object)


def is_pixel_an_extremum(first_subimage, second_subimage, third_subimage, threshold):
    """if the center element of 3x3x3 array > neighbors ,return True, else False"""
    # 检测尺度空间的局部极值点，所有图像的大小是相同的，只是模糊量sigma不同
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            """检测center_pixel是否大于上一张,下一张,第一行,第三行,第二行第一个,第二行第三个的所有值，即26邻域"""
            return (
                np.all(center_pixel_value >= first_subimage)
                and np.all(center_pixel_value >= third_subimage)
                and np.all(center_pixel_value >= second_subimage[0, :])
                and np.all(center_pixel_value >= second_subimage[2, :])
                and center_pixel_value >= second_subimage[1, 0]
                and center_pixel_value >= second_subimage[1, 2]
            )
        elif center_pixel_value < 0:
            return (
                np.all(center_pixel_value <= first_subimage)
                and np.all(center_pixel_value <= third_subimage)
                and np.all(center_pixel_value <= second_subimage[0, :])
                and np.all(center_pixel_value <= second_subimage[2, :])
                and center_pixel_value <= second_subimage[1, 0]
                and center_pixel_value <= second_subimage[1, 2]
            )
        return False


def localize_extremum_via_quadratic_fit(
    i,
    j,
    image_index,
    octave_index,
    num_intervals,
    dog_images_in_octave,
    sigma,
    contrast_threshold,
    image_boarder_width,
    eigenvalue_ratio=10,
    num_attempts_until_convergence=5,
):
    """iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors"""
    # 沿 宽度、高度、比例三个维度将极值点定位到子像素级别
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(
        num_attempts_until_convergence
    ):  # 迭代五次，超过迭代次数的x被舍弃
        # convert uint8 to float32 to compute derivatives and normalization
        first_image, second_image, third_image = dog_images_in_octave[
            image_index - 1 : image_index + 2
        ]
        pixel_cube = (
            np.stack(
                [
                    first_image[i - 1 : i + 2, j - 1 : j + 2],
                    second_image[i - 1 : i + 2, j - 1 : j + 2],
                    third_image[i - 1 : i + 2, j - 1 : j + 2],
                ]
            ).astype("float32")
            / 255
        )  # 归一化
        gradient = compute_gradient_at_center_pixel(pixel_cube)
        hessian = compute_hessian_matrix_at_center_pixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[
            0
        ]  # 线性最小二乘更新极值点，这里求出来的是offset，i,j,image_index是初始极值点在尺度空间下的三维坐标
        if (
            abs(extremum_update[0]) < 0.5
            and abs(extremum_update[1]) < 0.5
            and abs(extremum_update[2]) < 0.5
        ):  # x中的所有元素小于0.5时结束迭代
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))  # 根据offset修正
        image_index += int(round(extremum_update[2]))
        # make sure pixel cube in the image
        if (
            i < image_boarder_width
            or i >= image_shape[0] - image_boarder_width
            or j < image_boarder_width
            or j >= image_shape[1] - image_boarder_width
            or image_index < 1
            or image_index > num_intervals
        ):
            extremum_is_outside_image = True
            break
    # 超越图片边界或者超越迭代上限就直接退出
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    function_value_at_updated_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(
        gradient, extremum_update
    )
    if (
        abs(function_value_at_updated_extremum) * num_intervals >= contrast_threshold
    ):  # 更新极值点函数值，剔除弱响应。即响应的绝对值x尺度数需要大于threshold，否则该点被剔除
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if (
            xy_hessian_det > 0
            and eigenvalue_ratio * (xy_hessian_trace**2)
            < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det
        ):  # 边缘响应剔除，即hessian矩阵行列式为负，或者 k * tr(H) - (k+1) ** 2 * det(H) < 0 (其实就是harris response要大于0, 小于0说明特征值一大一小，为边缘，或者两个都很小，为flat region）这个点为边缘 or 平滑区域，被舍弃。
            keypoint = cv2.KeyPoint()
            keypoint.pt = (
                (j + extremum_update[0]) * (2**octave_index),
                (i + extremum_update[1]) * (2**octave_index),
            )
            keypoint.octave = (
                octave_index
                + image_index * (2**8)
                + int(np.round((extremum_update[2] + 0.5) * 255)) * (2**16)
            )
            keypoint.size = (
                sigma
                * (
                    2
                    ** ((image_index + extremum_update[2]) / np.float32(num_intervals))
                )
                * (2 ** (octave_index + 1))
            )
            # octave_index + 1 because the input image was doubled
            keypoint.response = abs(function_value_at_updated_extremum)
            return keypoint, image_index
    return None


def compute_gradient_at_center_pixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size"""
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h), 导数的定义式
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


def compute_hessian_matrix_at_center_pixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size"""
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # 以上为Hessian矩阵计算公式
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (
        pixel_array[1, 2, 2]
        - pixel_array[1, 2, 0]
        - pixel_array[1, 0, 2]
        + pixel_array[1, 0, 0]
    )
    dxs = 0.25 * (
        pixel_array[2, 1, 2]
        - pixel_array[2, 1, 0]
        - pixel_array[0, 1, 2]
        + pixel_array[0, 1, 0]
    )
    dys = 0.25 * (
        pixel_array[2, 2, 1]
        - pixel_array[2, 0, 1]
        - pixel_array[0, 2, 1]
        + pixel_array[0, 0, 1]
    )
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


def find_scale_space_extrema(
    gaussian_images,
    dog_images,
    num_intervals,
    sigma,
    image_boarder_width,
    contrast_threshold=0.04,
):
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
            zip(
                dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:]
            )
        ):
            # (i, j) is the center of the 3x3 array
            for i in range(
                image_boarder_width, first_image.shape[0] - image_boarder_width
            ):
                for j in range(
                    image_boarder_width, first_image.shape[1] - image_boarder_width
                ):  # 遍历当前DoG图像的每个像素，避免边界
                    if is_pixel_an_extremum(
                        first_image[i - 1 : i + 2, j - 1 : j + 2],
                        second_image[i - 1 : i + 2, j - 1 : j + 2],
                        third_image[i - 1 : i + 2, j - 1 : j + 2],
                        threshold,
                    ):  # 检测当前像素是否为局部极值点，如果是就对其进一步精确化定位
                        localization_result = localize_extremum_via_quadratic_fit(
                            i,
                            j,
                            image_index + 1,
                            octave_index,
                            num_intervals,
                            dog_images_in_octave,
                            sigma,
                            contrast_threshold,
                            image_boarder_width,
                        )  # image_index + 1是因为在dog_images_in_octave里实际要检测第二幅图像
                        if localization_result is not None:
                            keypoint, localized_image_index = (
                                localization_result  # 如果能精确化定位，就获取这个点和所在的图像层索引
                            )
                            keypoints_with_orientations = (
                                compute_keypoints_with_orientations(
                                    keypoint,
                                    octave_index,
                                    gaussian_images[octave_index][
                                        localized_image_index
                                    ],
                                )
                            )  # 计算这个点的方向，这样就得到了keypoint的位置和方向
                            for (
                                keypoint_with_orientation
                            ) in keypoints_with_orientations:
                                keypoints.append(
                                    keypoint_with_orientation
                                )  # 可能有多个方向，因此为每个方向创建并附加一个新的关键点。
    return keypoints


"""
    关键点的位置 ( keypoint.pt ) 根据其图层重复加倍，以便它对应于基础图像中的坐标。 keypoint.octave和keypoint.size （即比例）属性是根据OpenCV 实现定义的。
"""

"""
    出现是局部极值点但无法精确定位的情况：
        1.边界效应：即点太靠近边界，像素不足以进行梯度计算和hessian矩阵计算，这个通过image_border_width避免
        2.低对比度：极值点容易受到图像噪声的影响，即极值性质可能是由于噪声而非实际特征得到，这个通过弱响应剔除解决
        3.高主曲率：极值点在边缘上，这个通过hessian矩阵剔除边缘响应得到解决
        4.迭代不收敛：这个通过设置最大迭代次数解决。
    方向计算失败的情况：
        1.梯度方向不明显：在平坦区域或噪声较大的区域，方向直方图没有明显峰值，导致计算失败
        2.局部信息不足：在图像边界会遇到这种情况，已经在定位部分解决。
"""


def compute_keypoints_with_orientations(
    keypoint,
    octave_index,
    gaussian_image,
    radius_factor=3,
    num_bins=36,
    peak_ratio=0.8,
    scale_factor=1.5,
):
    """计算keypoints的 orientation,为keypoint的邻域像素创建梯度直方图"""
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # 计算尺度，半径，权重因子；初始化直方图用于存储方向的统计信息
    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))# scale用于确定关键点周围区域的大小，以便计算梯度方向和幅值，因为图像在不同的octave上有不同的尺度，每个octave上原图相当于被放大一倍，因此相应的邻域范围也要缩小一倍，这样代表的区域就不变，以此获得尺度不变性。
    radius = int(np.round(radius_factor * scale))# 决定像素周围的邻域大小，这里为 3 * scale， 由于高斯分布99.7%的概率密度位于3个标准差内，因此 3 * scale的像素拥有99.7%的权重
    weight_factor = -0.5 / (scale**2)#高斯分布中常数项
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(
        -radius, radius + 1
    ):  # 遍历keypoint周围一定半径的像素，region_x,region_y是当前像素的坐标
        region_y = int(np.round(keypoint.pt[1] / np.float32(2**octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = (
                    int(np.round(keypoint.pt[0] / np.float32(2**octave_index))) + j
                )
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = (
                        gaussian_image[region_y, region_x + 1]
                        - gaussian_image[region_y, region_x - 1]
                    )
                    dy = (
                        gaussian_image[region_y + 1, region_x]
                        - gaussian_image[region_y - 1, region_x]
                    )  # 计算x,y方向梯度并用高斯滤波降噪
                    gradient_magnitude = np.sqrt(dx**2 + dy**2)  # 计算梯度幅值
                    gradient_orientation = np.rad2deg(
                        np.arctan2(dy, dx)
                    )  # 计算梯度朝向
                    weight = np.exp(
                        weight_factor * (i**2 + j**2)
                    )  # 通过高斯权重因子来加权当前像素的梯度大小，使靠近keypoint的像素对直方图贡献更大
                    histogram_index = int(
                        np.round(gradient_orientation * num_bins / 360.0)
                    )  # 将梯度方向从角度（0-360）归一化，再转换为bin的索引（0-36）
                    raw_histogram[histogram_index % num_bins] += (
                        weight * gradient_magnitude
                    )  # 将梯度幅值映射到每个对应bin中

    for n in range(num_bins):  # 平滑直方图
        smooth_histogram[n] = (
            6 * raw_histogram[n]
            + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins])
            + raw_histogram[n - 2]
            + raw_histogram[(n + 2) % num_bins]
        ) / 16.0  # 6 x 当前bin + 4 x 左右bin + 1 x 左右第二个bin （6+ 4x2 + 1x2）=16所以最后除以16归一化， 以上系数对应的是5点高斯滤波器
    orientation_max = np.max(smooth_histogram)
    # np.roll将直方图平移，目的是将原始直方图的值与其左右的值进行比较，找到局部peak，logical_and检查是否大于左右,where返回满足条件的索引，即peak的index
    orientation_peaks = np.where(
        np.logical_and(
            smooth_histogram > np.roll(smooth_histogram, 1),
            smooth_histogram > np.roll(smooth_histogram, -1),
        )
    )[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]  # 获取每个index对应的峰值
        if (
            peak_value >= peak_ratio * orientation_max
        ):  # 过滤弱峰值，只考虑大于主峰值一定比例的峰值
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            # 通过二次插值计算更精确的峰值索引，因为直方图index是离散的，真实的峰值可能在离散点之间的小数点上，通过二次插值来找到真实峰值对应的小数点。公式在上面网址
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (
                peak_index
                + 0.5
                * (left_value - right_value)
                / (left_value - 2 * peak_value + right_value)
            ) % num_bins
            orientation = (
                360.0 - interpolated_peak_index * 360.0 / num_bins
            )  # 计算插值后索引对应的角度，用360度减去复原后的角度是因为sift算法为了确保方向定义的一致性，计算的方向是从正X轴的逆时针方向计算，即从右往左倒着算。
            if abs(orientation - 360.0) < float_tolerance:  # 控制数值溢出
                orientation = 0
            new_keypoint = cv2.KeyPoint(
                *keypoint.pt,#坐标
                keypoint.size,#邻域大小
                orientation, #主方向
                keypoint.response, #方向强度
                keypoint.octave, #在哪个octave和layer里
            )  # 创建keypoint对象
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

'''清理重复的keypoint'''
def compare_keypoints(keypoint1, keypoint2):
    '''比较两个关键点的顺序，返回一个整数表示相对顺序，如果keypoint1响应比keypoint2小，return true'''
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def remove_duplicate_keypoints(keypoints):
    if len(keypoints) < 2:
        return keypoints
    
    keypoints.sort(key=cmp_to_key(compare_keypoints))#按照坐标从小到大排序，对于同一点，保留size，angle(较小),response,octave大的点
    unique_keypoints = [keypoints[0]]
    
    for next_keypoint in keypoints[1:]:#如果后一个keypoint的x,y,size,angle都与前一个不同，添加进列表
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
               unique_keypoints.append(next_keypoint)
    return unique_keypoints

# keypoint 尺度变换
def convert_keypoints_to_input_image_size(keypoints):
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))#将关键点的坐标缩小一半，keypoint.pt类型是tuple,必须先转换为np数组才能进行操作，最后变换回来
        keypoint.size *= 0.5 #大小也缩小一半
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & ~255)#将octave与 255进行按位与操作，清零低8位数据，-1后重复此操作，再按位或合并两者。高8位是octave里的layer_index，低8位是octave_index,清0低8位直接将octave映射到octave 0，即原始图像
        converted_keypoints.append(keypoint)

    return converted_keypoints

# 生成描述符
def unpack_octave(keypoint):
    ''' 计算keypoint的 octave, layer和 scale '''
    octave = keypoint.octave & 255 #octave_index
    layer = (keypoint.octave >> 8) & 255 # layer_index
    if octave >= 128:
        octave = octave | -128 # octave大于等于128时，视为负数修正
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    
    return octave, layer, scale

def descriptor_generation(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    '''将每个keypoint周围的区域划分为 4x4的子块（window_width ** 2），对每个子块进行8个方向(num_bins)的直方图统计 '''
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360. #8个方向
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))# (x,y)左右各加1是为了处理边界效应

        # Descriptor window size (described by half_width) follows OpenCV convention  还有这种传统... -_-
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size #对不同scale采用不同的窗口尺寸
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))#公式 r= sqrt(1/2) * w *(d+1) d是轴上的子区域数量，子区域为 4*4， 乘根号二是因为对角线长度
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))#确保窗口半径不超过图像边界

        # 对每个keypoint，将坐标系变换到与其orientation有关的局部坐标系，确保了旋转不变性
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle# y坐标
                col_rot = col * cos_angle - row * sin_angle# x坐标
                # 归一化旋转后的坐标(即计算相对位置)
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5 
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:#确保坐标在有效范围内
                    # 转换为实际图像坐标
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    # 确保坐标在图像边界内
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        # 计算梯度幅值和朝向
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row + 1, window_col] - gaussian_image[window_row - 1, window_col]
                        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2)) # 对梯度模长进行高斯加权,离得近的权重大
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # 用 inverse trilinear interpolation插值来平滑梯度直方图，将中心点的值分布到8个邻域点，简而言之，思想就是根据离中心点的距离按比例分配中心点的值，离越近分配越多，先把中心点1分为2，再2分4,4分8
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1,(orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010 
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2,(orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten() # 去掉histogram的边界，展平为128-D的tensor
        # 设置阈值及归一化到单位长度
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold #超过上限（0.2）的值被截断
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance) # 再次归一化到0-255
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        # map到0-255间
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0 
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append (descriptor_vector)
    return np.array(descriptors, dtype='float32')


def drawKeypointsWithoutOrientation(image, keypoints, color=(0, 0, 255)):
    """
    在图片上绘制没有方向的关键点
    :param image: 输入图片
    :param keypoints: 关键点列表
    :param color: 关键点的颜色（默认红色，BGR格式）
    :return: 绘制了关键点的图像
    """
    keypoints_without_orientation = [
    cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints
    ]
    return cv2.drawKeypoints(
            image,
            keypoints_without_orientation,
            None,
            color,
            cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )


def sort_descriptors_by_keypoints(descriptors, keypoints, strongest_keypoints):
    """
    根据关键点排序描述符
    :param descriptors: 描述符列表
    :param keypoints: 所有关键点列表
    :param strongest_keypoints: 最强的关键点列表
    :return: 根据 strongest_keypoints 排序后的描述符列表
    """
    sorted_descriptors = []
    for strong_kp in strongest_keypoints:
        for i, kp in enumerate(keypoints):
            # 比较关键点的位置和大小，以确保匹配
            if (strong_kp.pt == kp.pt) and (strong_kp.size == kp.size):
                sorted_descriptors.append(descriptors[i])
                break
    return sorted_descriptors


def plot_descriptor(descriptors, img, keypoints):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    # plt.scatter(keypoint.pt[0],keypoint.pt[1], c="r")  # 关键点位置

    num_bins = 8
    bin_width = 360 / num_bins

    # 描述符是128维的向量
    n = 10
    for descriptor, keypoint in zip(descriptors[::n], keypoints[::n]):
        x, y = keypoint.pt[0], keypoint.pt[1]
        for i in range(0, len(descriptor), num_bins):
            for j in range(num_bins):
                angle = bin_width * j
                magnitude = descriptor[i + j]
                dx = magnitude * np.cos(np.deg2rad(angle))
                dy = magnitude * np.sin(np.deg2rad(angle))
                plt.arrow(x, y, dx, dy, color="b", head_width=1)
                x += 4
                if (j + 1) % 4 == 0:  # 为了模拟4x4的网格换行
                    y += 4
                    x -= 16
        # 恢复原始位置，避免下一个keypoint的位置偏移
        x, y = keypoint.pt[0], keypoint.pt[1]


def computeKeypointsAndDescriptors(
    image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5
):
    """Compute SIFT keypoints and descriptors for an input image"""
    image = image.astype("float32")
    base_image = generate_base_image(image, sigma, assumed_blur)
    num_octaves = compute_num_of_octaves(base_image.shape)
    gaussian_kernels = generate_Gaussian_kernels(sigma, num_intervals)
    gaussian_images = generate_Gaussian_images(base_image, num_octaves, gaussian_kernels)
    dog_images = generate_DoG_images(gaussian_images)
    keypoints = find_scale_space_extrema(
        gaussian_images, dog_images, num_intervals, sigma, image_border_width
    )
    keypoints = remove_duplicate_keypoints(keypoints)
    keypoints = convert_keypoints_to_input_image_size(keypoints)
    descriptors = descriptor_generation(keypoints, gaussian_images)
    return keypoints, descriptors


if __name__ == '__main__':
    with Timer():
        sigma = 1.6
        assumed_blur = 0.5
        num_intervals = 3
        image_boarder_width = 5
        image = cv2.imread("ac1329024dce07bc6cbb31f1796e59bb_r.jpg", cv2.IMREAD_GRAYSCALE)
        # image = cv2.imread('xuan_gou.jpg', cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32")
        base_image = generate_base_image(image, sigma, assumed_blur)
        plt.imshow(np.clip(base_image, 0, 255).astype(np.uint8), cmap="gray")
        plt.title("base image")
        plt.axis("off")
        plt.show()
        num_octaves = compute_num_of_octaves(base_image.shape)
        Gaussian_kernels = generate_Gaussian_kernels(sigma, num_intervals)
        Gaussian_images = generate_Gaussian_images(base_image, num_octaves, Gaussian_kernels)
        DoG_images = generate_DoG_images(Gaussian_images)

        for octave_index in range(num_octaves):
            num_images_in_octave = len(Gaussian_images[octave_index])
            num_dog_images_in_octave = len(DoG_images[octave_index])
            fig, axes = plt.subplots(1, num_images_in_octave, figsize=(20, 5))
            fig_dog, axes_dog = plt.subplots(1, num_dog_images_in_octave, figsize=(15, 5))
            for img_index in range(num_images_in_octave):
                gaussian_img_display = np.clip(
                    Gaussian_images[octave_index][img_index], 0, 255
                ).astype(np.uint8)

                axes[img_index].imshow(gaussian_img_display, cmap="gray")
                axes[img_index].set_title(f"Octave {octave_index} : Gaussian_Image {img_index}")
                axes[img_index].axis("off")
                    
            for dog_img_index in range(num_dog_images_in_octave):
                dog_img_display = np.clip(
                    DoG_images[octave_index][dog_img_index], 0, 255
                ).astype(np.uint8)

                axes_dog[dog_img_index].imshow(dog_img_display, cmap="gray")
                axes_dog[dog_img_index].set_title(
                    f"Octave {octave_index} : DoG_Image {dog_img_index}"
                )
                axes_dog[dog_img_index].axis("off")
                
        # plt.tight_layout()
        plt.show()

        keypoints = find_scale_space_extrema(
            Gaussian_images, DoG_images, num_intervals, sigma, image_boarder_width
        )
        
        image_with_keypoints = drawKeypointsWithoutOrientation(image.astype('uint8'), keypoints)
        cv2.imwrite("image_with_keypoints.jpeg", image_with_keypoints)
        cv2.imshow("image_with_keypoints", image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        keypoints = remove_duplicate_keypoints(keypoints)
        keypoints = convert_keypoints_to_input_image_size(keypoints)
        image_with_converted_keypoints = drawKeypointsWithoutOrientation(image.astype('uint8'), keypoints)
        cv2.imwrite("image_with_converted_keypoints.jpeg", image_with_converted_keypoints)
        cv2.imshow("image_with_converted_keypoints", image_with_converted_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        image_with_conv_keypoints_ori = cv2.drawKeypoints(image.astype('uint8'), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('image_with_kp_and_ori.jpeg', image_with_conv_keypoints_ori)
        cv2.imshow('image_with_kp_and_ori.jpeg', image_with_conv_keypoints_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        sorted_keypoints = sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True) #降序排列
        strongest_250 = sorted_keypoints[:250]
        strongest_500 = sorted_keypoints[:500]
        image_with_strongest250 = drawKeypointsWithoutOrientation(image.astype('uint8'), strongest_250)
        image_with_strongest500 = drawKeypointsWithoutOrientation(image.astype('uint8'), strongest_500)
        cv2.imwrite("strongest_250_points.jpeg", image_with_strongest250)
        cv2.imshow("strongest_250_points", image_with_strongest250)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("strongest_500_points.jpeg", image_with_strongest500)
        cv2.imshow("strongest_500_points", image_with_strongest500)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for keypoint in keypoints:
            octave, layer, scale = unpack_octave(keypoint)
        descriptors = descriptor_generation(keypoints, Gaussian_images)
        print(descriptors.shape)

        # 根据前250个关键点重新排序描述符
        sorted_descriptors_250 = sort_descriptors_by_keypoints(descriptors, keypoints, strongest_250)
        plot_descriptor(sorted_descriptors_250, image.astype('uint8'), strongest_250)
        plt.savefig('strongest_250_descriptors.jpeg')
        plt.show()
