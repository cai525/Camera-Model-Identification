import cv2 as cv
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np


def LBP_preprocess(img):
    hsv = np.array(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    hsv[:, :, 2] = local_binary_pattern(hsv[:, :, 2], 16, 2)
    return hsv


if __name__ == '__main__':
    path = '../datasets/Train/Apple_iPhone6Plus/train_543_6.png'
    img = plt.imread(path)
    img_new = LBP_preprocess(img)
    plt.imshow(img_new[:,:,2])
    plt.show()
