from .model import dengNet
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ContfidenceLoss(nn.Module):

    def __init__(self):  # margin=2
        super(ContfidenceLoss, self).__init__()

    def forward(self, logits):
        loss = 0
        for l in logits:
            p = F.log_softmax(l)
            index = torch.max(p, 0)[1]
            y_i = torch.exp(l[index]) / (torch.exp(l[0]) + torch.exp(l[1]))
            loss += (-1) * torch.log(y_i)

        return loss

def calGrad(data,pt_path):
    device = torch.device("cuda")
    model = dengNet.Net().to(device)
    all_state = torch.load(pt_path)
    model.load_state_dict(all_state['original_state'],strict=True)
    model.eval()
    data.requires_grad = True

    output = model(data)
    criterion=ContfidenceLoss().cuda()
    loss=criterion(output)

    model.zero_grad()
    loss.requires_grad_()
    loss.backward()
    grad = data.grad.data
    return abs(grad)
