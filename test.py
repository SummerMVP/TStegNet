import random
import time
import sys
import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import utils
from data import build_train_loader
from data import build_val_loader
from model import TStegNet,gradient,loss,modules
import os


test_cover_dir="/data2/mingzhihu/dataset/MAE_suni4/test/cover"
test_stego_dir="/data2/mingzhihu/dataset/ADVEMB_mipod02/test/stego"#change

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'


# test_cover_dir="F:/dataset/ADVEMB_suni04/cover"
# test_stego_dir="F:/dataset/ADVEMB_suni04/stego"

ck_path='/data2/mingzhihu/code/SiaStegNet-master/src/final_model/Sialwe_mipod2/model_best.pth' #change
pt_path="/data2/mingzhihu/code/SiaStegNet-master/src/final_model/param_mipod/mipod2_params.pt" #change
utils.set_random_seed(5708292)
net = TStegNet.Net()
net.cuda()
net = nn.DataParallel(net)
device = torch.device('cuda:0')
net.to(device)
# net = nn.DataParallel(net)#训练时加入，这里也要加
net.load_state_dict(torch.load(ck_path)['state_dict'], strict=True)

val_loader = build_val_loader(
    test_cover_dir, test_stego_dir, batch_size=32,
    num_workers=2
)
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = modules.ContrastiveLoss(margin=1)
criterion_1.cuda()
criterion_2.cuda()


def preprocess_data_new(images, labels):
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]

    ch, cw, h0, w0 = h, w, 0, 0
    cw = cw & ~1
    grad=gradient.calGrad(images.cuda(), pt_path)
    images=images.numpy()
    temp1=images[..., h0:h0 + ch//3, w0:w0 + cw // 2].copy()
    images[..., h0:h0 + ch//3, w0:w0 + cw // 2]=images[..., h0 + ch//3:h0 + ch*2 // 3, w0:w0 + cw // 2]
    images[..., h0 + ch//3:h0 + ch * 2 // 3, w0:w0 + cw // 2]=temp1

    temp2 = images[..., h0:h0 + ch // 3, w0 + cw // 2:w0 + cw].copy()
    images[..., h0:h0 + ch // 3, w0 + cw // 2:w0 + cw] = images[..., h0 + ch//3:h0 + ch * 2 // 3, w0 + cw // 2:w0 + cw]
    images[..., h0 + ch//3:h0 + ch * 2 // 3, w0 + cw // 2:w0 + cw] = temp2
    images=torch.from_numpy(images)

    inputs = [
        images[..., h0:h0 + ch, w0:w0 + cw // 2],
        images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
    ]
    inputs = [x.cuda() for x in inputs]
    labels = labels.cuda()
    grad = grad.cuda()
    return inputs, labels,grad

def test():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    for data in val_loader:
        inputs, labels,grad = preprocess_data_new(data['image'], data['label'])
        with torch.no_grad():
            outputs, euclidean_distance = net(inputs,grad)
            valid_loss += criterion_1(outputs, labels).item() + \
                          0.1 * criterion_2(euclidean_distance, labels)
            valid_accuracy += modules.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    print('Test set: Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy

if __name__=='__main__':
    test()