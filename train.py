from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import utils
from data import build_train_loader
from data import build_val_loader
from model import TStegNet,gradient,loss,modules
from visdom import Visdom
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-cover-dir', dest='train_cover_dir', type=str, required=False,
        default="/data2/mingzhihu/dataset/MAE_suni4/train/cover",
    )
    parser.add_argument(
        '--val-cover-dir', dest='val_cover_dir', type=str, required=False,
        default="/data2/mingzhihu/dataset/MAE_suni4/val/cover",
    )
    parser.add_argument(
        '--train-stego-dir', dest='train_stego_dir', type=str, required=False,
        default="/data2/mingzhihu/dataset/MAE_suni2/train/mixStego",
    )
    parser.add_argument(
        '--val-stego-dir', dest='val_stego_dir', type=str, required=False,
        default="/data2/mingzhihu/dataset/MAE_suni2/val/mixStego",
    )

    parser.add_argument('--epoch', dest='epoch', type=int, default=200)  # default=1000
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4)
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
    parser.add_argument('--margin', dest='margin', type=float, default=1.00)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)#3张卡用40
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=2)#3张卡用3

    parser.add_argument('--finetune', dest='finetune', type=str, default=None)
    parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--seed', dest='seed', type=int, default=3407)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=875)
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, required=False,
                        default="model_data/Sialwe_mix2suni_stego")
    parser.add_argument('--pt_path', dest='pt_path', type=str, required=False,
                        default="final_model/param_suni/suni2_params.pt")
    parser.add_argument('--lr-strategy', dest='lr_str', type=int, default=2,
                        help='1: StepLR, 2:MultiStepLR, 3:ExponentialLR, 4:CosineAnnealingLR, 5:ReduceLROnPlateau')

    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)

    args.cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    log_file = osp.join(args.ckpt_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=0)

    utils.set_random_seed(None if args.seed < 0 else args.seed)

    logger.info('Command Line Arguments: {}'.format(str(args)))

args = parse_args()
setup(args)

logger.info('Building data loader')

train_loader, epoch_length = build_train_loader(
    args.train_cover_dir, args.train_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
val_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
train_loader_iter = iter(train_loader)

logger.info('Building model')

net = TStegNet.Net()



criterion_1 = nn.CrossEntropyLoss()
criterion_2 = loss.ContrastiveLoss(margin=args.margin)

if args.cuda:
    net.cuda()
    net = nn.DataParallel(net)
    device = torch.device('cuda:0')
    net.to(device)
    criterion_1.cuda()
    criterion_2.cuda()

if args.finetune is not None:
    print("loaded!")
    net.load_state_dict(torch.load(args.finetune)['state_dict'], strict=True)





optimizer = Adamax(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)

lr_str = args.lr_str
if lr_str == 1:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
elif lr_str == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200],
                                                     gamma=0.1)  # milestones=[900,975]
elif lr_str == 3:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
elif lr_str == 4:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
elif lr_str == 5:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3,
                                                           patience=10, verbose=True, min_lr=0,
                                                           eps=1e-08)
else:
    raise NotImplementedError('Unsupported learning rate strategy')


def preprocess_data(images, labels):
    # images of shape: NxCxHxW
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]

    ch, cw, h0, w0 = h, w, 0, 0

    cw = cw & ~1
    grad = gradient.calGrad(images.cuda(),args.pt_path)
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

    if args.cuda:
        inputs = [x.cuda() for x in inputs]
        labels = labels.cuda()
        grad = grad.cuda()
    return inputs, labels,grad


def train(epoch):
    net.train()
    running_loss, running_accuracy = 0., 0.

    for batch_idx in range(epoch_length):
        data = next(train_loader_iter)
        inputs, labels,grad = preprocess_data(data['image'], data['label'])

        optimizer.zero_grad()
        outputs, euclidean_distance = net(inputs, grad)
        loss = criterion_1(outputs, labels) + \
               args.alpha * criterion_2(euclidean_distance, labels)
        accuracy = modules.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    running_accuracy /= args.log_interval
    running_loss /= args.log_interval
    logger.info(
        'Train epoch: {} [{}/{}] Accuracy: {:.2f}% Loss: {:.6f}'.format(
            epoch, batch_idx + 1, epoch_length, 100 * running_accuracy,
            running_loss))
    return running_loss, running_accuracy


def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.

    for data in val_loader:
        inputs, labels,grad = preprocess_data(data['image'], data['label'])
        with torch.no_grad():
            outputs, euclidean_distance = net(inputs,grad)
            valid_loss += criterion_1(outputs, labels).item() + \
                          args.alpha * criterion_2(euclidean_distance, labels).item()
            valid_accuracy += modules.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    logger.info('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


_time = time.time()
best_accuracy = 0.
# 可视化
# viz = Visdom()
# viz.line([0.], [0], win="loss", opts=dict(title='Loss'))
# viz.line([0.], [0], win="accuracy", opts=dict(title='Accuracy'))
for e in range(1, args.epoch + 1):
    logger.info('Epoch: {}'.format(e))
    logger.info('Train')
    train_loss,train_accu=train(e)
    logger.info('Time: {}'.format(time.time() - _time))
    logger.info('Test')
    loss, accuracy = valid()
    # viz.line(X=[e], Y=[train_loss], win="loss", name='train_loss', update='append')
    # viz.line(X=[e], Y=[loss], win="loss", name='valid_loss', update='append')
    # viz.line(X=[e], Y=[train_accu], win="accuracy", name='train_accuracy', update='append')
    # viz.line(X=[e], Y=[accuracy], win="accuracy", name='valid_accuracy', update='append')
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(accuracy)
    else:
        scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        is_best = True
    else:
        is_best = False
    logger.info('Best accuracy: {}'.format(best_accuracy))
    logger.info('Time: {}'.format(time.time() - _time))
    save_checkpoint(
        {
            'epoch': e,
            'state_dict': net.state_dict(),
            'best_prec1': accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth'))
    # if(e%20==0):
    #     save_checkpoint(
    #         {
    #             'epoch': e,
    #             'state_dict': net.state_dict(),
    #             'best_prec1': accuracy,
    #             'optimizer': optimizer.state_dict(),
    #         },
    #         False,
    #         filename=os.path.join(args.ckpt_dir, 'model_'+str(e)+'.pth'),
    #         best_name=os.path.join(args.ckpt_dir, 'model_best.pth'))
