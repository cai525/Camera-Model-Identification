"""
Step4:
    以85%的质量因子压缩图片
"""
import os
from PIL import Image
import numpy as np
from random import sample

if __name__ == "__main__":
    parent_dir = "..\\datasets\\Train"    # <------原父目录 --------->
    type_list = os.listdir(parent_dir)
    # 选取图片加噪
    for img_type in type_list:
        img_dir = os.listdir(os.path.join(parent_dir, img_type))  # 图像目录
        # <------设定抽样数--------->
        # num = len(img_dir)  # 设定为目录大小
        num = 300  # 设定为90
        # <------------ 如果是对训练集操作，记得剔除加噪声的图像--------------->
        new_dir = []
        # 将名字不包含noise的图片名加入新集合
        for name in img_dir:
            if name.find('noise') == -1:
                new_dir.append(name)
        imgName_list = sample(new_dir, num)  # 抽样
        for i, imgName in enumerate(imgName_list):
            # 指定来源和去向
            src = os.path.join(os.path.join(parent_dir, img_type, imgName))
            new_parent_dir = "..\\datasets\\Train"  # <------目标目录--------->
            # 读图像，压缩保存
            img = Image.open(src)
            # <------设置质量因子--------->
            q = 85
            img.save(os.path.join(new_parent_dir,img_type, 'compress{0}.jpg'.format(i)), quality=q)
