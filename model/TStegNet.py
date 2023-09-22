import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import os,math
from .modules import *
from .MPNCOV import *


# SRM_npy = np.load('SRM_Kernels.npy')
SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))
class L2_nrom(nn.Module):
    def __init__(self,mode='l2'):
        super(L2_nrom, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True)).pow(0.5)
            norm = embedding / (embedding.pow(2).mean(dim=1, keepdim=True)).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x)
            embedding = _x.sum((2,3), keepdim=True)
            norm = embedding / (torch.abs(embedding).mean(dim=1, keepdim=True))
        return norm

class Sepconv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Sepconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)


    def forward(self, input):

        out1 = self.conv1(input)
        out = self.conv2(out1)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           num_input_features,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, num_input_features,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, prev_features):
        new_features = self.conv1(self.relu1(self.norm1(prev_features)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # self.add_module('pool', Statispooling(kernel_size=2, stride=2))

class _DenseBlock_Add(nn.Module):
    def __init__(self, num_layers, num_input_features):
        super(_DenseBlock_Add, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = init_features
        for name, layer in self.named_children():
            new_features = layer(features)
            features = features + new_features
        return features



class DenseNet_Add_1(nn.Module):
    def __init__(self, num_layers=6):
        super(DenseNet_Add_1, self).__init__()

        # 高通滤波 卷积核权重初始化
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=False)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy

        self.features = nn.Sequential(OrderedDict([('norm0', nn.BatchNorm2d(30)), ]))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        block = _DenseBlock_Add(
            num_layers=num_layers,
            num_input_features=30,)
        self.features.add_module('denseblock%d' % 1, block)#preprocessing

        num_features = 30

        trans = _Transition(num_input_features=num_features,
                            num_output_features=32)   # BlockB
        self.features.add_module('transition%d' % 1, trans)

    def forward(self, input):
        HPF_output = F.conv2d(input, self.srm_filters_weight, stride=1, padding=2)
        output = self.features(HPF_output)
        # output=Statispooling(output, 4, 4)
        return output

class Net(nn.Module):
    def __init__(self,p=0.5):
        super(Net, self).__init__()
        #preprocessing+BlockB
        self.Dense_layers = DenseNet_Add_1(num_layers=6)

        # feature extraction
        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # BlockC
        self.layer5_BN = nn.BatchNorm2d(32)
        self.layer5_AC = nn.ReLU()

        self.layer6 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # BlockC
        self.layer6_BN = nn.BatchNorm2d(64)
        self.layer6_AC = nn.ReLU()

        self.avgpooling2 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # BlockC
        self.layer7_BN = nn.BatchNorm2d(64)
        self.layer7_AC = nn.ReLU()

        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # BlockC
        self.layer8_BN = nn.BatchNorm2d(128)
        self.layer8_AC = nn.ReLU()

        self.avgpooling3 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # BlockC
        self.layer9_BN = nn.BatchNorm2d(128)
        self.layer9_AC = nn.ReLU()

        self.layer10 = Sepconv(128, 128)  # BlockD
        self.layer10_BN = nn.BatchNorm2d(128)
        self.layer10_AC = nn.ReLU()
        # MGP
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.L2_norm = L2_nrom(mode='l2')
        self.L1_norm = L2_nrom(mode='l1')
        # classifier 384
        self.fc = nn.Linear(128 * 3*4+1+128, 2)
        self.dropout = nn.Dropout(p=p)

        norm_layer = nn.BatchNorm2d
        self.srm = SRMConv2d(1, 0)
        self.bn1 = norm_layer(30)
        self.relu = nn.ReLU(inplace=True)
        self.Layer1 = PartA(30, 64, norm_layer=norm_layer)
        self.Layer2 = LayerA(64, 16, norm_layer=norm_layer)
        self.Layer3 = LayerA(16, 16, norm_layer=norm_layer)

        self.Layer4 = LayerB(16, 16, norm_layer=norm_layer)
        self.Layer5 = LayerB(16, 16, norm_layer=norm_layer)
        self.Layer6 = LayerB(16, 16, norm_layer=norm_layer)
        self.Layer7 = LayerB(16, 16, norm_layer=norm_layer)

        self.Layer8 = LayerC(16, 16, norm_layer=norm_layer)
        self.Layer9 = LayerC(16, 64, norm_layer=norm_layer)
        self.Layer10 = LayerC(64, 128, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_feat(self, input):

        Dense_block_out = self.Dense_layers(input)
        layer5_out = self.layer5(Dense_block_out)
        layer5_out = self.layer5_BN(layer5_out)
        layer5_out = self.layer5_AC(layer5_out)

        layer6_out = self.layer6(layer5_out)
        layer6_out = self.layer6_BN(layer6_out)
        layer6_out = self.layer6_AC(layer6_out)

        avg_pooling2 = self.avgpooling2(layer6_out)
        # avg_pooling2 = Statispooling(layer6_out, 4, 4)

        layer7_out = self.layer7(avg_pooling2)
        layer7_out = self.layer7_BN(layer7_out)
        layer7_out = self.layer7_AC(layer7_out)

        layer8_out = self.layer8(layer7_out)
        layer8_out = self.layer8_BN(layer8_out)
        layer8_out = self.layer8_AC(layer8_out)

        avg_pooling3 = self.avgpooling2(layer8_out)
        # avg_pooling3 = Statispooling(layer8_out, 4, 4)

        layer9_out = self.layer9(avg_pooling3)
        layer9_out = self.layer9_BN(layer9_out)
        layer9_out = self.layer9_AC(layer9_out)

        layer10_out = self.layer10(layer9_out)
        layer10_out = self.layer10_BN(layer10_out)
        layer10_out = self.layer10_AC(layer10_out)
        output_GAP = self.GAP(layer10_out)
        output_L2 = self.L2_norm(layer10_out)
        output_L1 = self.L1_norm(layer10_out)
        output_GAP = output_GAP.view(-1, 128)
        output_L2 = output_L2.view(-1, 128)
        output_L1 = output_L1.view(-1, 128)
        Final_feat = torch.cat([output_GAP, output_L2, output_L1], dim=-1)
        return Final_feat

    def extract_grad_feat(self, x):

        x = x.float()
        out = self.srm(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.Layer1(out)
        out = self.Layer2(out)
        out = self.Layer3(out)
        out = self.Layer4(out)
        out = self.Layer5(out)
        out = self.Layer6(out)
        out = self.Layer7(out)
        out = self.Layer8(out)
        out = self.Layer9(out)
        out = self.Layer10(out)

        # output = CovpoolLayer(out)
        # output = SqrtmLayer(output, 5)
        # out = TriuvecLayer(output)
        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1))

        return out

    def forward(self, args,grad):
        ############# statistics fusion start #############
        feats = torch.stack(
            [self.extract_feat(subarea) for subarea in args], dim=0
        )
        grad_feat = self.extract_grad_feat(grad)

        euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6,
                                                 keepdim=True)

        if feats.shape[0] == 1:
            final_feat = feats.squeeze(dim=0)
        else:
            # feats_sum = feats.sum(dim=0)
            # feats_sub = feats[0] - feats[1]
            feats_mean = feats.mean(dim=0)
            feats_var = feats.var(dim=0)
            feats_min, _ = feats.min(dim=0)
            feats_max, _ = feats.max(dim=0)

            '''feats_sum = feats.sum(dim=0)
            feats_sub = abs(feats[0] - feats[1])
            feats_prod = feats.prod(dim=0)
            feats_max, _ = feats.max(dim=0)'''

            # final_feat = torch.cat(
            #    [feats[0], feats[1], feats[0], feats[1]], dim=-1
            #    #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            # )

            final_feat = torch.cat(
                [euclidean_distance, feats_mean, feats_var, feats_min, feats_max,grad_feat], dim=-1
                # [euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            )

        out = self.dropout(final_feat)
        # out = self.fcfusion(out)
        # out = self.relu(out)
        out = self.fc(out)

        return out, euclidean_distance




if __name__ == '__main__':
    from torchsummary import summary
    Input = torch.randn(1, 1, 256, 256).cuda()
    net = TStegNet().cuda()
    print(summary(net,(1,256,256)))