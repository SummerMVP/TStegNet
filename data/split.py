# coding=utf-8
import os, random, shutil
import scipy.misc


# 将图片拆分成训练集train(0.8)和验证集val(0.2)
dir1='train/stego'
dir2='val/stego'
dir3='test/stego'
def moveFile(Dir, train_ratio=0.5, val_ratio=0.1,test_ratio=0.4):
    if not os.path.exists(os.path.join(Dir, dir1)):
        os.makedirs(os.path.join(Dir, dir1))

    if not os.path.exists(os.path.join(Dir, dir2)):
        os.makedirs(os.path.join(Dir, dir2))

    if not os.path.exists(os.path.join(Dir, dir3)):
        os.makedirs(os.path.join(Dir, dir3))

    filenames = []
    for root, dirs, files in os.walk(Dir):
        for name in files:
            filenames.append(name)
        break

    filenum = len(filenames)

    num_train = int(filenum * train_ratio)
    num_val = int(filenum * val_ratio)
    # num_test = int(filenum * test_ratio)
    # sample_train = random.sample(filenames, num_train)
    p=0
    for name in filenames:
        if(p<num_train):
            shutil.copy(os.path.join(Dir, str(p+1)+'.pgm'), os.path.join(Dir, dir1))
            p+=1
        elif(p<num_train+num_val):
            shutil.copy(os.path.join(Dir, str(p+1)+'.pgm'), os.path.join(Dir, dir2))
            p+=1
        else:
            shutil.copy(os.path.join(Dir, str(p+1)+'.pgm'), os.path.join(Dir, dir3))
            p+=1

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
    Dir =r'F:\dataset\data\suni_0.4\output_same\stego'
    moveFile(Dir)
    print("处理完成")
    # break
