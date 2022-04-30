import os,shutil
if __name__ == "__main__":
    des_dir = '../datasets/Val_main'
    src1 = '../datasets/Val_noise'
    src2 = '../datasets/Val_compress'
    typeList = os.listdir(des_dir)

    for src in src1,src2:
        for camera_type in typeList:
            type_path = os.path.join(src,camera_type)
            img_list = os.listdir(type_path)
            for i in range(35):
                shutil.copy(os.path.join(type_path,img_list[i]),
                            os.path.join(des_dir,camera_type))
