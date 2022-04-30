"""
    前面高斯噪声方差设置错了，得删了重加，压缩也不太对，把加高斯噪声的图片也压缩了，呜呜呜。。。
    该程序是用于从数据集里删除验证集的
"""
import os
import re
Val_path = "..\\datasets\\Val"
val_set = set()
# 用正则匹配验证集数据
for types in os.listdir(Val_path):
    for img_name in os.listdir(os.path.join(Val_path,types)):
        val_set.add(re.search(r'train_\d+',img_name).group(0)+'.JPG')
print(val_set)
# 从训练集删除数据
train_path = "..\\datasets\\Train"
for types in os.listdir(train_path):
    for img_name in os.listdir(os.path.join(train_path,types)):
        if img_name in val_set:
            os.remove(os.path.join(train_path,types,img_name))
            print('remove '+img_name)
# # 检查
# train_path = "..\\datasets\\Train"
# for types in os.listdir(train_path):
#     for img_name in os.listdir(os.path.join(train_path,types)):
#         if img_name in val_set:
#             print("ERROR")