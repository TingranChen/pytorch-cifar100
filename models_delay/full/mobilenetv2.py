"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import utils_mine.DelayExpansion as DEX

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100, delay_data=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.delay_data = delay_data
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.delay_layer = DEX.DelayExpansionLayer(delay_data=self.delay_data)

    def forward(self, x):
        in_data = x
        for layer in self.residual:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.delay_layer(x, x.size(0), layer)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x += in_data

        return x

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, delay_data=None):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.delay_layer = delay_data

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, delay_data=self.delay_layer)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6, delay_data=self.delay_layer)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)
        self.delay_layer = DEX.DelayExpansionLayer(delay_data=self.delay_layer)

    def forward(self, x):

        for layer in self.pre:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.delay_layer(x, x.size(0), layer)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)

        for layer in self.conv1:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.delay_layer(x, x.size(0), layer)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        self.delay_layer(x, x.size(0), self.conv2)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, delay_data=self.delay_layer))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, delay_data=self.delay_layer))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2():
    """
    创建VGG模型的通用函数，支持不同VGG配置。
    :param model_type: 模型类型，例如 'A', 'B', 'D', 'E'
    :param delay_data: 延时膨胀参数数据
    :param batch_norm: 是否使用批归一化
    :return: VGG模型实例
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
    return MobileNetV2(delay_data = delay_data)