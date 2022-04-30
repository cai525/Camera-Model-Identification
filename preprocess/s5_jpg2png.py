"""
Step5:
    将jpg转存为png格式（因为测试集是png格式）
"""
import cv2 as cv
import os


def jpg2png(old_path:str):
    img = cv.imread(old_path)
    new_path = old_path.replace('jpg','png')
    cv.imwrite(new_path,img)
    os.remove(old_path)


if __name__ == "__main__":
    typeList = os.listdir("..\\datasets\\Train")
    for path1 in ['..\\datasets\\Train']:
        for path2 in typeList:
            # 获得所有图片文件名
            parent_path = os.path.join(path1,path2)
            imgName_list = os.listdir(parent_path)
            for imgName in imgName_list:
                jpg2png(os.path.join(parent_path,imgName))

