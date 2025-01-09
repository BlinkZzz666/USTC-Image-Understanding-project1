import cv2
from PythonSIFT import pysift

image = cv2.imread("ac1329024dce07bc6cbb31f1796e59bb_r.jpg", 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
print(descriptors.shape)
output_image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow('keypoints', output_image)
cv2.waitKey(0)