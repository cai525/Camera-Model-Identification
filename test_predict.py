"""
用于测试标签是否贴的正确
"""
from models.vgg_256 import Vgg_256
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':

    # 索引是model定义中的标号，索引对应的数字是要求的编号
    labelList = [7,  # 'Apple_iPhone6Plus',
                 1,  # 'Canon_PowerShotA640',
                 3,  # 'Sony_DSC-W170',
                 9,  # 'Samsung_GalaxyS5',
                 4,  # 'Huawei_P9',
                 2,  # 'Nikon_D70s',
                 10,  # 'OnePlus_A3003',
                 6,  # 'Microsoft_Lumia640LTE',
                 5,  # 'Lenovo_P70A',
                 8  # 'Xiaomi_RedmiNote3']
                 ]
    num = 30  # 两个测试集中的照片数目
    # # ========================== 导入预测模型 ======================== #
    Vgg256 = Vgg_256()
    # 编译
    Vgg256.model.load_weights("model_weight/vgg256/vgg256[p22][93.59].hdf5")
    Vgg256.compile()
    # ========================== 读取excel表格 ======================== #
    excel = pd.read_excel('测试用.xlsx')
    excel.insert(2,'Prob',0)
    excel.insert(3, 'Pro2', 0)
    excel.insert(4, 'Prob3', 0)
    # ========================== 预测验证集1 ======================== #
    print('\n =========== 测试用 begin =========== ')
    for i in range(num):
        img_path = os.path.join(r'datasets\测试用', excel.iloc[i, 0])
        pro = Vgg256.predict(img_path)
        ind = np.argmax(pro)  # 最大概率对应的下标
        label = labelList[ind]
        pro.sort()
        excel.iloc[i, 1] = label
        excel.iloc[i, 2] = pro.max()
        excel.iloc[i, 3] = pro[-2]
        excel.iloc[i, 4] = pro[-3]
        if i % 10 == 0:
            print('>', end='')
    print('\n =========== 测试用 end =========== ')

    # ======================== 保存 ================================== #
    excel.to_excel("测试预测结果.xlsx")
