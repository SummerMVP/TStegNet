# coding=utf-8
import os, random, shutil
import scipy.misc


# 将图片拆分成训练集train(0.8)和验证集val(0.2)
# dir1='/data2/mingzhihu/dataset/MAE_suni2/train/stego'
# dir2='/data2/mingzhihu/dataset/MAE_suni2/val/stego'
# dir3='/data2/mingzhihu/dataset/MAE_suni2/test/stego'
def moveFile(Dir, train_ratio=0.7, val_ratio=0.1,test_ratio=0.2):
    dir1=Dir+'/train/stego'
    dir2=Dir+'/val/stego'
    dir3=Dir+'/test/stego'
    if not os.path.exists(dir1):
        os.makedirs(dir1)

    if not os.path.exists(dir2):
        os.makedirs(dir2)

    if not os.path.exists(dir3):
        os.makedirs(dir3)

    filenames = []
    for root, dirs, files in os.walk(Dir):
        for name in files:
            filenames.append(name)
        break

    filenum = len(filenames)

    num_train = int(filenum * train_ratio)
    num_val = int(filenum * val_ratio)+num_train
    # num_test = int(filenum * test_ratio)
    # sample_train = random.sample(filenames, num_train)
    p=0
    for name in filenames:
        p=int(name.split('.')[0])
        if(p<num_train+1):
            shutil.move(os.path.join(Dir, name), dir1)
            p+=1
        elif(p<num_val+1):
            shutil.move(os.path.join(Dir, name), dir2)
            p+=1
        else:
            shutil.move(os.path.join(Dir, name), dir3)

    # sample_val = list(set(filenames).difference(set(sample_train)))
    #
    # for name in sample_val:
    #     shutil.move(os.path.join(Dir, name), os.path.join(Dir, 'val_cover'))


if __name__ == '__main__':
    # Dir = input('请输入想处理的文件夹：')
    # for root,dirs,files in os.walk(Dir):
    #     for name in dirs:
    #         folder = os.path.join(root, name)
    #         print("正在处理:" + folder)
    Dir =r'/data2/mingzhihu/dataset/MAE_mipod2'
    moveFile(Dir)
    print("处理完成")
    # break
