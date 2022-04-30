"""
    功能：用于快速在datesets文件夹中创建目录
"""
import os
if __name__ == "__main__":
    typeList = os.listdir('../datasets/Train')
    parent_dir = "../datasets/Val_noise"
    dir_list = [os.path.join(parent_dir,file_type) for file_type in typeList]
    for file_dir in dir_list:
        os.makedirs(file_dir)
