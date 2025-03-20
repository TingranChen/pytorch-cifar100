"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import pandas as pd
import utils_mine.DelayCalculation as DEC  # 假设你有这个模块

# ... 其他代码 ...

class DelayCalculationConv2d(nn.Conv2d):
    def __init__(self, delay_layer, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DelayCalculationConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayCalculationConv2d, self).forward(input)  # output = Conv2d.forward(input)
        batch_size = output.size(0)
        assert output.dim() == 4
        self.delay_layer(input, batch_size, self)  # Conv2d(input)
        return output

class DelayCalculationLinear(nn.Linear):
    def __init__(self, delay_layer, in_features, out_features, bias=True):
        super(DelayCalculationLinear, self).__init__(in_features, out_features, bias)
        self.delay_layer = delay_layer

    def forward(self, input):
        output = super(DelayCalculationLinear, self).forward(input)
        batch_size = output.size(0)
        assert output.dim() == 2
        self.delay_layer(output, batch_size, self)
        return output

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, delay_layer, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            DelayCalculationConv2d(delay_layer, in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayCalculationConv2d(delay_layer, out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                DelayCalculationConv2d(delay_layer, in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, delay_layer, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            DelayCalculationConv2d(delay_layer, in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayCalculationConv2d(delay_layer, out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DelayCalculationConv2d(delay_layer, out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                DelayCalculationConv2d(delay_layer, in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, delay_data, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.delay_layer = DEC.DelayCalculationLayer()

        self.conv1 = nn.Sequential(
            DelayCalculationConv2d(self.delay_layer, 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, self.delay_layer, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, self.delay_layer, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, self.delay_layer, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, self.delay_layer, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DelayCalculationLinear(self.delay_layer, 512 * block.expansion, num_classes)

    def _make_layer(self, block, delay_layer, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(delay_layer, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

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
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(delay_data, BasicBlock,[2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(delay_data, BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(delay_data, BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(delay_data, BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(delay_data, BottleNeck, [3, 8, 36, 3])