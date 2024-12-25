"""residual attention network in pytorch



[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import utils_mine.DelayExpansion as DEX  # 假设你有这个模块

#"""The Attention Module is built by pre-activation Residual Unit [11] with the
#number of channels in each stage is the same as ResNet [10]."""


class DelayExpansionConv2d(nn.Conv2d):
    def __init__(self, delay_layer, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DelayExpansionConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayExpansionConv2d, self).forward(input)
        batch_size = output.size(0)
        if output.dim() == 4:
            self.delay_layer(output, batch_size, self)
        return output

class DelayExpansionLinear(nn.Linear):
    def __init__(self, delay_layer, in_features, out_features, bias=True):
        super(DelayExpansionLinear, self).__init__(in_features, out_features, bias)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayExpansionLinear, self).forward(input)
        batch_size = output.size(0)
        if output.dim() == 2:
            self.delay_layer(output, batch_size, self)
        return output

class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, downsample the featuremap
    """
    def __init__(self, delay_layer, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = int(out_channels / 4)
        self.residual_function = nn.Sequential(
            #1x1 conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, in_channels, bottleneck_channels, 1, stride),
            #3x3 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, bottleneck_channels, bottleneck_channels, 3, padding=1),
            #1x1 conv
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, bottleneck_channels, out_channels, 1)
        )
        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels):
            self.shortcut = DelayExpansionConv2d(delay_layer, in_channels, out_channels, 1, stride=stride)

    def forward(self, x):

        res = self.residual_function(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class AttentionModule1(nn.Module):
    def __init__(self, delay_layer, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(delay_layer, in_channels, out_channels, p)
        self.trunk = self._make_residual(delay_layer, in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown4 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.shortcut_short = PreActResidualUnit(delay_layer, in_channels, out_channels, 1)
        self.shortcut_long = PreActResidualUnit(delay_layer, in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(delay_layer, in_channels, out_channels, p)

    def forward(self, x):
        ###We make the size of the smallest output map in each mask branch 7*7 to be consistent
        #with the smallest trunk output map size.
        ###Thus 3,2,1 max-pooling layers are used in mask branch with input size 56 * 56, 28 * 28, 14 * 14 respectively.
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 28
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #28 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut_long = self.shortcut_long(x_s)

        #seccond downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #14 shortcut
        shape2 = (x_s.size(2), x_s.size(3))
        shortcut_short = self.soft_resdown3(x_s)

        #third downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown3(x_s)

        #mid
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        #thrid upsample out 54
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, delay_layer, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(delay_layer, in_channels, out_channels, 1))
        return nn.Sequential(*layers)

class AttentionModule2(nn.Module):

    def __init__(self, delay_layer, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(delay_layer, in_channels, out_channels, p)
        self.trunk = self._make_residual(delay_layer, in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(delay_layer, in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(delay_layer, in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #14 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut = self.shortcut(x_s)

        #seccond downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        #mid
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, delay_layer, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(delay_layer, in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule3(nn.Module):

    def __init__(self, delay_layer, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(delay_layer, in_channels, out_channels, p)
        self.trunk = self._make_residual(delay_layer, in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(delay_layer, in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(delay_layer, in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(delay_layer, in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(delay_layer, in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #mid
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, delay_layer, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(delay_layer, in_channels, out_channels, 1))

        return nn.Sequential(*layers)
class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """

    def __init__(self, block_num, delay_data, class_num=100):
        super().__init__()
        self.delay_layer = DEX.DelayExpansionLayer(delay_data)
        self.pre_conv = nn.Sequential(
            DelayExpansionConv2d(self.delay_layer, 3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(self.delay_layer, 64, 256, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(self.delay_layer, 256, 512, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(self.delay_layer, 512, 1024, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            PreActResidualUnit(self.delay_layer, 1024, 2048, 2),
            PreActResidualUnit(self.delay_layer, 2048, 2048, 1),
            PreActResidualUnit(self.delay_layer, 2048, 2048, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.linear = DelayExpansionLinear(self.delay_layer, 2048, class_num)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_stage(self, delay_layer, in_channels, out_channels, num, block):
        layers = []
        layers.append(PreActResidualUnit(delay_layer, in_channels, out_channels, 2))
        for _ in range(num):
            layers.append(block(delay_layer, out_channels, out_channels))
        return nn.Sequential(*layers)

data = [
    [0.0, 0.056598642],
    [0.0666667, 0.205962435],
    [0.1333333, 0.312138982],
    [0.2, 0.437158198],
    [0.2666667, 0.319973934],
    [0.3333333, 0.450264408],
    [0.4, 0.559485637],
    [0.4666667, 0.694916383],
    [0.5333333, 0.562896787],
    [0.6, 0.709107365],
    [0.6666667, 0.811728286],
    [0.7333333, 0.939352112],
    [0.8, 0.818508719],
    [0.8666667, 0.958645411],
    [0.9333333, 1.072683293],
    [1.0, 1.192973781]
]
# 将数据转换为 DataFrame
delay_data = pd.DataFrame(data, columns=["data", "delay expension"])
def attention56():
    return Attention([1, 1, 1], delay_data)

def attention92():
    return Attention([1, 2, 3], delay_data)

