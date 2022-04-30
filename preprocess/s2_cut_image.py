"""
Step2：分割为512×512的区域
     分割方式：左上(lu)，左下(ld)，右上角(ru)，右下角(rd)，中间(m)
             中上（mu）,中下（md）,中左（ml），中右（mr）
"""
from os import listdir, remove
from random import seed, randrange
import numpy as np
import cv2 as cv
import os.path
import matplotlib.pyplot as plt


def divide_img(path, ind):
    img = np.array(cv.imread(path))
    # 读完即删除原图
    remove(path)
    # ----------------- 裁剪为四份 -------------------------
    # 计算图片大小
    row, col = img.shape[:2]
    if row < 512 or col < 512:
        return
    new_path = path.replace('.jpg', '')
    cv.imwrite(new_path + "_lu_{}.jpg".format(ind), img[0:512, 0:512, :])
    # 左下
    cv.imwrite(new_path + "_ld_{}.jpg".format(ind), img[-512:, 0:512, :])
    # 右上
    cv.imwrite(new_path + "_ru_{}.jpg".format(ind), img[0:512, -512:, :])
    # 右下
    cv.imwrite(new_path + "_rd_{}.jpg".format(ind), img[-512:, -512:, :])
    # 正中间
    mr, mc = row // 2, col // 2
    cv.imwrite(new_path + "_m_{}.jpg".format(ind), img[mr - 256:mr + 256, mc - 256:mc + 256, :])
    # 左中间
    cv.imwrite(new_path + "_lm_{}.jpg".format(ind), img[mr - 256:mr + 256, 0:512, :])
    # 右中间
    cv.imwrite(new_path + "_rm_{}.jpg".format(ind), img[mr - 256:mr + 256, -512:, :])
    # 上中间
    cv.imwrite(new_path + "_um_{}.jpg".format(ind), img[0:512, mc - 256:mc + 256, :])
    # 下中间
    cv.imwrite(new_path + "_dm_{}.jpg".format(ind), img[-512:, mc - 256:mc + 256, :])


if __name__ == "__main__":
    train_path = "../datasets/Train"
    val_path = "../datasets/Val"
    type_list = listdir(train_path)
    for path1 in train_path, val_path:
        for type_name in type_list:
            img_list = listdir(os.path.join(path1, type_name))
            for i, img_name in enumerate(img_list):
                # 读取图片
                path = os.path.join(path1, type_name, img_name)
                divide_img(path, i)
            print(path1 + type_name + "------完成")
    # divide_img("../test_img.jpg", 1)
