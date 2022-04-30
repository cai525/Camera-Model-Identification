"""
 定义了类vgg_256 —— 输入512*512的图片后将进行裁剪，变成256*256的子集，即网络的输入是256*256*3；
 预测时通过原图片的四个部分结果加权获得最终的概率
"""

# 加载tensorflow模型
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
from random import randrange
import sys

# 定义宏参数
PICS_WIDTH, PICS_HEIGHT = 256, 256  # 注意大小是256*256
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRIC = 'categorical_accuracy'
NUM_CATEGS = 10


class Vgg_256:
    """
    实现和vgg_256相关的一系列操作，但注意对vgg模型训练的设置需要手动设置vgg_256.model属性
    """

    def __init__(self):
        """
        初始化模型-注意：不加载权重，只是创造self.model的基本结构
        """
        # ######################### 定义手机种类  ################################## #
        self.category = ['Apple_iPhone6Plus', 'Canon_PowerShotA640',
                         'Sony_DSC-W170', 'Samsung_GalaxyS5',
                         'Huawei_P9', 'Nikon_D70s',
                         'OnePlus_A3003', 'Microsoft_Lumia640LTE',
                         'Lenovo_P70A', 'Xiaomi_RedmiNote3']

        # ####################### 定义学习率等参数 ###################################### #
        self.lr = 0.001  # 学习率
        self.loss = 'categorical_crossentropy'
        self.metrics = 'categorical_accuracy'

        # ############################# 定义模型  ###################################### #
        #   ========================== 卷积层-vgg16 =====================================
        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling='avg')
        #   ========================= 全连接层 =========================================
        flat1 = Flatten()(base_model.layers[-1].output)
        # 第一层
        class1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat1)
        dropout1 = Dropout(0.2)(class1)
        # 输出层
        output = Dense(NUM_CATEGS, activation='softmax')(dropout1)
        self.model = Model(inputs=base_model.inputs, outputs=output)

    def compile(self):
        # 编译模型
        model_optimizers = optimizers.SGD(lr=self.lr, momentum=0.8, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=model_optimizers, metrics=[self.metrics])

    def gen(self, directory, batch_size):
        """
        指定目录的数据生成器
        :param directory: 数据集目录
        :param batch_size: 批量大小
        :return: 随机剪裁的256*256的图片
        """
        data_gen = ImageDataGenerator()
        train_it = data_gen.flow_from_directory(directory=directory, target_size=(512, 512),
                                                classes=self.category, class_mode="categorical",
                                                batch_size=batch_size)
        while True:
            X, y = next(train_it)
            X = preprocess_input(X)
            a, b = randrange(0, 256), randrange(0, 256)
            yield X[:, a:a + 256, b:b + 256, :], y

    def predict(self, img_path):
        """
        用于预测输入的图片
        输入：图片路径
        输出：图片类别的评价softmax分类预测
        """
        # 读入图片（keras的坑爹机制决定了必须这样读）
        img = load_img(img_path)
        img = img_to_array(img)
        if img.shape != (512, 512, 3):
            print('Image size error!')
            sys.exit(1)

        # 预处理（非常必要，千万别忘了）
        img = preprocess_input(img)
        # 提供子图片的左上角坐标
        originPoint = [[slice(0, 256), slice(0, 256)],
                       [slice(0, 256), slice(256, 512)],
                       [slice(256, 512), slice(0, 256)],
                       [slice(256, 512), slice(256, 512)]
                       ]
        prob = np.zeros(10)
        for p in originPoint:
            sub = img[p[0], p[1]]
            prob += (self.model.predict(np.expand_dims(sub, axis=0))).reshape(-1) * 0.25

        return prob
