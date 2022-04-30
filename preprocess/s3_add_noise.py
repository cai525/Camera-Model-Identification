"""
Step3:
    添加高斯白噪声
    根据观察，噪声方差设置为5-25之间的随机数应当比较合理
"""
import os
import cv2 as cv
import numpy as np
from random import sample, randrange
import matplotlib.pyplot as plt


def add_gaussian_noise(img: np.ndarray, std):
    shape = img.shape[0:2]
    for i in range(3):
        noise = np.random.normal(0, std, shape)
        noise = noise.round()
        temp = img[:, :, i] + noise
        temp = np.clip(temp, a_min=0, a_max=255)
        img[:, :, i] = temp.astype('uint8')


if __name__ == "__main__":
    # ===================================== #
    parent_dir = "..\\datasets\\Train"
    # ===================================== #
    type_list = os.listdir(parent_dir)
    # 选取90张图片加噪
    for img_type in type_list:
        img_dir = os.listdir(os.path.join(parent_dir, img_type))  # 图像目录
        # num = len(img_dir)  # 目录大小
        num = 300  # 加噪数目
        imgName_list = sample(img_dir, num)  # 抽样
        for i, imgName in enumerate(imgName_list):
            # 指定来源和去向
            src = os.path.join(os.path.join(parent_dir, img_type, imgName))
            # =================================================== #
            new_parent_dir = "..\\datasets\\Train"  # 目标目录
            # =================================================== #
            # 读图像，加噪（方差范围5-26）
            img = cv.imread(src)
            std = randrange(10, 26)
            # =================================================== #
            add_gaussian_noise(img, std)
            cv.imwrite(os.path.join(new_parent_dir, img_type, "noise{0}.jpg").format(i), img)

    # img = cv.imread('../img.jpg')
    # add_gaussian_noise(img, 25)
    # plt.imshow(img[:, :, [2, 1, 0]])
    # plt.show()
