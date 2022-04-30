"""
=======================    数据导入   ===========================================
因为图片存在硬盘中，所以我们需要将图片导入内存，来进行后续操作。数据集大小为3GB，我的电脑内存为16GB，
因而这样的操作没有问题。

在该模块我们实现的功能：
1. 将训练集和验证集转为矩阵格式
2. 将图片和标签关联
3. 打乱顺序
4. 保存为npz格式，方便下次读取
"""
from os import listdir
from os.path import join
import numpy as np
import sklearn
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# load all images into memory
def load_dataset(dataset_path):
    """
    输入：数据集（训练或者测试集）路径
    输出：打乱后的训练集和对应标签
    """
    images, labels = list(), list()
    # 获取类别(** 类别均已Train文件夹下的为准 **)
    types = listdir('datasets\\Train')

    for i, camera_type in enumerate(types):
        # 创建每一类的one-hot label(y)
        label = np.zeros(len(types))
        label[i] = 1
        imgName_list = listdir(join(dataset_path, camera_type))
        for imgName in imgName_list:
            # ------------- 读取照片 ----------------- #
            img_path = join(dataset_path, camera_type, imgName)
            img = load_img(img_path)
            img = img_to_array(img, dtype='uint8')
            # ---> check point: 返回图片大小 (512, 512, 3)
            # ------------- 添加进集合 --------------- #
            images.append(img)
            labels.append(label)

    # 转成narray
    X = np.asarray(images, dtype='uint8')
    Y = np.asarray(labels, dtype='uint8')
    # 随机打乱
    X, Y = sklearn.utils.shuffle(X, Y)
    return X,Y


if __name__ == '__main__':
    # <----------- 输入文件夹 --------->
    dataset_path = "datasets\\Val_compress"
    # 调用函数得到对应目录的X、Y
    X,Y = load_dataset(dataset_path)
    # save both arrays
    # <----------- 输出文件夹 --------->
    save_name = "datasets\\Val_compress.npz"
    np.savez(save_name, X, Y)
