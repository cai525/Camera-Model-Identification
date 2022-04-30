"""
    Step1:用于在最开始将训练集和验证集分隔开
    【必须在预处理前分开，保证训练集和验证集完全无关】
"""
from os import listdir
from random import randrange
import shutil
import os.path
if __name__=="__main__":
    # 最初所有图片都存在训练集
    train_path = "../datasets/Train"
    val_path = "../datasets/Val"
    type_names = listdir(train_path)
    # 每一类移走10张作为验证集
    move_num = 10
    for type_name in type_names:
        for _ in range(move_num):
            # 每一次都更新目录，防止抽中被移走的图
            img_names = listdir(os.path.join(train_path, type_name))
            # 从目录中随机抽取一张图片
            img_name = img_names[randrange(0,len(img_names))]
            # 移动
            src = os.path.join(train_path,type_name,img_name)
            des = os.path.join(val_path,type_name,img_name)
            shutil.move(src,des)





