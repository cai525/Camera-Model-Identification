"""
Step2：分割为512×512的区域
     分割方式：左上(lu)，左下(ld)，右上角(ru)，右下角(rd)，中间(m)
             中上（mu）,中下（md）,中左（ml），中右（mr）
    New:
    随机分割30份
"""
from os import listdir, remove
from random import seed, randrange
import numpy as np
import cv2 as cv
import os.path
import matplotlib.pyplot as plt


def divide_img(path):
    img = np.array(cv.imread(path))
    # 读完即删除原图
    remove(path)
    # ----------------- 裁剪为四份 -------------------------
    # 计算图片大小
    row, col = img.shape[:2]
    for i in range(30):
        px, py = randrange(0, row - 520), randrange(0, col - 520)
        new_path = path.replace('.jpg', "_{0}.jpg".format(i))
        new_path = new_path.replace('.JPG', "_{0}.jpg".format(i))
        cv.imwrite(new_path, img[px:px + 512, py:py + 512, :])


if __name__ == "__main__":
    train_path = "../datasets/Train"
    type_list = listdir(train_path)
    for path1 in [train_path]:
        for type_name in type_list:
            img_list = listdir(os.path.join(path1, type_name))
            for img_name in img_list:
                # 读取图片
                path = os.path.join(path1, type_name, img_name)
                divide_img(path)
            print(path1 + type_name + "------完成")
    # divide_img("../test_img.jpg", 1)
