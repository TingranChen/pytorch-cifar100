"""dense net in pytorch



[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn
import pandas as pd
import utils_mine.DelayExpansion as DEX  # 假设你有这个模块

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

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, delay_layer, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            DelayExpansionConv2d(delay_layer, inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        self.delay_layer = delay_layer

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, delay_layer, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            DelayExpansionConv2d(delay_layer, in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        self.delay_layer = delay_layer

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, delay_data, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.delay_layer = DEX.DelayExpansionLayer(delay_data)
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        self.conv1 = DelayExpansionConv2d(self.delay_layer, 3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential()
        for index in range(len(nblocks) - 1):
            self.features.add_module(
                "dense_block_layer_{}".format(index),
                self._make_dense_layers(block, inner_channels, nblocks[index])
            )
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(index),
                Transition(self.delay_layer, inner_channels, out_channels)
            )
            inner_channels = out_channels
        self.features.add_module(
            "dense_block{}".format(len(nblocks) - 1),
            self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1])
        )
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = DelayExpansionLinear(self.delay_layer, inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module(
                'bottle_neck_layer_{}'.format(index),
                block(self.delay_layer, in_channels, self.growth_rate)
            )
            in_channels += self.growth_rate
        return dense_block

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
def densenet121(delay_data = delay_data):
    return DenseNet(delay_data, Bottleneck, [6, 12, 24, 16], growth_rate=32)

def densenet169(delay_data = delay_data):
    return DenseNet(delay_data, Bottleneck, [6, 12, 32, 32], growth_rate=32)

def densenet201(delay_data = delay_data):
    return DenseNet(delay_data, Bottleneck, [6, 12, 48, 32], growth_rate=32)

def densenet161(delay_data = delay_data):
    return DenseNet(delay_data, Bottleneck, [6, 12, 36, 24], growth_rate=48)

