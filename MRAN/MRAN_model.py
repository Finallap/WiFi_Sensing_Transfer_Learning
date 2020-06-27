import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch
import torch.nn.functional as F
import random


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MRANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(MRANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.Inception = InceptionA(2048, 64, num_classes)

    def forward(self, source, target, s_label):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        source, loss = self.Inception(source, target, s_label)

        return source, loss


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, num_classes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.source_fc = nn.Linear(288, num_classes)

    def forward(self, source, target, s_label):
        s_branch1x1 = self.branch1x1(source)

        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)

        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)

        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)

        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)

        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)

        t_branch1x1 = self.branch1x1(target)

        t_branch5x5 = self.branch5x5_1(target)
        t_branch5x5 = self.branch5x5_2(t_branch5x5)

        t_branch3x3dbl = self.branch3x3dbl_1(target)
        t_branch3x3dbl = self.branch3x3dbl_2(t_branch3x3dbl)
        t_branch3x3dbl = self.branch3x3dbl_3(t_branch3x3dbl)

        t_branch_pool = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        t_branch_pool = self.branch_pool(t_branch_pool)

        t_branch1x1 = self.avg_pool(t_branch1x1)
        t_branch5x5 = self.avg_pool(t_branch5x5)
        t_branch3x3dbl = self.avg_pool(t_branch3x3dbl)
        t_branch_pool = self.avg_pool(t_branch_pool)

        t_branch1x1 = t_branch1x1.view(t_branch1x1.size(0), -1)
        t_branch5x5 = t_branch5x5.view(t_branch5x5.size(0), -1)
        t_branch3x3dbl = t_branch3x3dbl.view(t_branch3x3dbl.size(0), -1)
        t_branch_pool = t_branch_pool.view(t_branch_pool.size(0), -1)

        source = torch.cat([s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool], 1)
        target = torch.cat([t_branch1x1, t_branch5x5, t_branch3x3dbl, t_branch_pool], 1)

        source = self.source_fc(source)
        t_label = self.source_fc(target)
        t_label = t_label.data.max(1)[1]

        loss = torch.Tensor([0])
        loss = loss.cuda()
        if self.training == True:
            loss += mmd.cmmd(s_branch1x1, t_branch1x1, s_label, t_label)
            loss += mmd.cmmd(s_branch5x5, t_branch5x5, s_label, t_label)
            loss += mmd.cmmd(s_branch3x3dbl, t_branch3x3dbl, s_label, t_label)
            loss += mmd.cmmd(s_branch_pool, t_branch_pool, s_label, t_label)
        return source, loss
