# 加载tensorflow模型
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

# 定义宏参数
PICS_WIDTH, PICS_HEIGHT = 512, 512
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRIC = 'categorical_accuracy'
NUM_CATEGS = 10


class Vgg16():

    def __init__(self, head_only):
        """
            head_only:选择是否只训练顶端（即自定义的全连接层）
            weights:选择是否从外部导入权重
            model:模型名称
            lr:学习率：默认为0.001
            """
        # ======================= 卷积层 ======================================
        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling='avg')
        # 是否训练头部
        if head_only:
            for lay in base_model.layers:
                lay.trainable = False
        # 将 0-14 层权重锁定，只训练最后一个block
        for i, lay in enumerate(base_model.layers):
            # print(i,lay)
            if i <= 14:
                lay.trainable = False

        # ======================= 全连接层 ======================================
        flat1 = Flatten()(base_model.layers[-1].output)
        # 第一层
        class1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat1)
        dropout1 = Dropout(0.2)(class1)
        # 第二层
        class2 = Dense(128, activation='relu', kernel_initializer='he_uniform')(dropout1)
        dropout2 = Dropout(0.2)(class2)
        # 输出层
        output = Dense(NUM_CATEGS, activation='softmax')(dropout2)
        self.model = Model(inputs=base_model.inputs, outputs=output)

        # define new model


