import torch
import torch.nn as nn
import torch.nn.functional as F


# cover 0 stego 1

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.25):  # margin=2
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        label = label.to(torch.float32)
        # euclidean_distance = F.pairwise_distance(output1, output2)#一般不会超过1
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +#载体
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)#载密
        )
        # print(loss_contrastive)
        # torch.pow是指数运算，如果是载密图像就不进行前一项，如果是载体图像就不进行后一项计算
        return loss_contrastive
