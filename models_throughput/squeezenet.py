"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import utils_mine.DelayExpansion as DEX

class DelayExpansionConv2d(nn.Conv2d):
    def __init__(self, delay_layer, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DelayExpansionConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # 对batch中每个样本计算各自的delay
        batch_size = output.size(0)
        assert output.dim() == 4
        self.delay_layer(output, batch_size, self)
        return output

class DelayExpansionLinear(nn.Linear):
    def __init__(self, delay_layer, in_features, out_features, bias=True):
        super(DelayExpansionLinear, self).__init__(in_features, out_features, bias)
        self.delay_layer = delay_layer
    def forward(self, input):
        output = super(DelayExpansionLinear, self).forward(input)
        batch_size = output.size(0)
        assert output.dim() == 2
        self.delay_layer(output, batch_size, self)
        return output

class Fire(nn.Module):

    def __init__(self, delay_layer, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            DelayExpansionConv2d(delay_layer, in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            DelayExpansionConv2d(delay_layer, squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            DelayExpansionConv2d(delay_layer, squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, delay_layer, class_num=100):

        super().__init__()
        self.stem = nn.Sequential(
            DelayExpansionConv2d(delay_layer, 3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(delay_layer, 96, 128, 16)
        self.fire3 = Fire(delay_layer, 128, 128, 16)
        self.fire4 = Fire(delay_layer, 128, 256, 32)
        self.fire5 = Fire(delay_layer, 256, 256, 32)
        self.fire6 = Fire(delay_layer, 256, 384, 48)
        self.fire7 = Fire(delay_layer, 384, 384, 48)
        self.fire8 = Fire(delay_layer, 384, 512, 64)
        self.fire9 = Fire(delay_layer, 512, 512, 64)

        self.conv10 = DelayExpansionConv2d(delay_layer, 512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x

def squeezenet(class_num=100):
    data = [
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
    delay_layer = DEX.DelayExpansionLayer(delay_data)
    return SqueezeNet(delay_layer, class_num=class_num)
