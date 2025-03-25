"""
resnet with stochastic depth

[1] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
    Deep Networks with Stochastic Depth

    https://arxiv.org/abs/1603.09382v3
"""
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import random
import utils_mine.DelayExpansion as DEX # 假设你已经有了这个模块
import pandas as pd

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
delay_data = pd.DataFrame(data, columns=["data", "delay expension"])
delay_layer = DEX.DelayExpansionLayer(delay_data)

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

class StochasticDepthBasicBlock(nn.Module):

    expansion=1

    def __init__(self, p, in_channels, out_channels, stride=1):
        super().__init__()
        self.p = p
        self.residual = nn.Sequential(
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, out_channels, out_channels * StochasticDepthBasicBlock.expansion, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * StochasticDepthBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                # 修改部分：nn.Conv2d -> DelayExpansionConv2d
                DelayExpansionConv2d(delay_layer, in_channels, out_channels * StochasticDepthBasicBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def survival(self):
        var = torch.bernoulli(torch.tensor(self.p).float())
        return torch.equal(var, torch.tensor(1).float().to(var.device))

    #@torch.jit.script_method
    def forward(self, x):
        if self.training:
            if self.survival():
                x = self.residual(x) + self.shortcut(x)
            else:
                x = self.shortcut(x)
        else:
            x = self.residual(x) * self.p + self.shortcut(x)

        return x

class StochasticDepthBottleNeck(nn.Module):

    expansion = 4

    def __init__(self, p, in_channels, out_channels, stride=1):
        super().__init__()

        self.p = p

        self.residual = nn.Sequential(
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, out_channels, out_channels * StochasticDepthBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * StochasticDepthBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * StochasticDepthBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                # 修改部分：nn.Conv2d -> DelayExpansionConv2d
                DelayExpansionConv2d(delay_layer, in_channels, out_channels * StochasticDepthBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * StochasticDepthBottleNeck.expansion)
            )

    #@torch.jit.script_method
    def forward(self, x):
        if self.training:
            if self.survival():
                x = self.residual(x) + self.shortcut(x)
            else:
                x = self.shortcut(x)
        else:
            x = self.residual(x) * self.p + self.shortcut(x)

        return x

class StochasticDepthResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            # 修改部分：nn.Conv2d -> DelayExpansionConv2d
            DelayExpansionConv2d(delay_layer, 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.step = (1 - 0.5) / (sum(num_block) - 1)
        self.pl = 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DelayExpansionLinear(delay_layer, 512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.pl, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            self.pl = max(0.5, self.pl - self.step)

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def stochastic_depth_resnet18(num_classes=100):
    return StochasticDepthResNet(StochasticDepthBasicBlock, [2, 2, 2, 2], num_classes)

def stochastic_depth_resnet34(num_classes=100):
    return StochasticDepthResNet(StochasticDepthBasicBlock, [3, 4, 6, 3], num_classes)

def stochastic_depth_resnet50(num_classes=100):
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 4, 6, 3], num_classes)

def stochastic_depth_resnet101(num_classes=100):
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 4, 23, 3], num_classes)

def stochastic_depth_resnet152(num_classes=100):
    return StochasticDepthResNet(StochasticDepthBottleNeck, [3, 8, 36, 3], num_classes)