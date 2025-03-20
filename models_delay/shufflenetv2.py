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
def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)
def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """
    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self, delay_layer, in_channels, out_channels, stride):
        super().__init__()
        self.delay_layer = delay_layer
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                DelayExpansionConv2d(self.delay_layer , in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                DelayExpansionConv2d(self.delay_layer , in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()
            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                DelayExpansionConv2d(self.delay_layer , in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x
        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)
        return x
class SHUFFLENET(nn.Module):
    def __init__(self, delay_data, ratio=1, class_num=100):
        super().__init__()
        self.delay_layer = DEX.DelayExpansionLayer(delay_data)
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')
        self.pre = nn.Sequential(
            DelayExpansionConv2d(self.delay_layer , 3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )
        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            DelayExpansionConv2d(self.delay_layer, out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )
        self.fc = DelayExpansionLinear(self.delay_layer, out_channels[3], class_num)
    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(self.delay_layer, in_channels, out_channels, 2))
        repeat -= 1
        while repeat:
            layers.append(ShuffleUnit(self.delay_layer, out_channels, out_channels, 1))
            repeat -= 1
        return nn.Sequential(*layers)
def shufflenetv2():
    """
    创建ShuffleNetV2模型的通用函数。
    :return: ShuffleNetV2模型实例
    """
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
    return SHUFFLENET(delay_data)