"""
    检验数据是否载入成功
    同时也提供了函数onehot2type，用于将onehot结果转成照相机的类别（str）
"""
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from sys import exit


def onehot2type(types: list, onehot: np.ndarray):
    ind = list(np.where(onehot == 1))
    if len(ind) != 1:
        print("类别数不为1")
        exit(1)
    ind = int(ind[0])
    return types[ind]


if __name__ == "__main__":
    data = np.load('datasets\\npz_file\\Val_main.npz')
    X, Y = data['arr_0'], data['arr_1']
    i = 100
    img = X[i, :, :, :]
    label = Y[i, :]
    types = listdir('datasets\\Train')
    print(onehot2type(types, label), ':', label)
    plt.imshow(img)
    plt.show()
