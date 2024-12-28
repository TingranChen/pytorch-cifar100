import torch.nn as nn
import pandas as pd
import utils_mine.DelayExpansion as DEX

# VGG 的配置参数
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


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


def make_layers(cfg, batch_norm=False, delay_layer=None):
    layers = []
    input_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = DelayExpansionConv2d(delay_layer, input_channel, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channel = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, delay_data, num_classes=100):
        super().__init__()
        self.delay_layer = DEX.DelayExpansionLayer(delay_data)
        self.features = make_layers(features, batch_norm=True, delay_layer=self.delay_layer)

        self.classifier = nn.Sequential(
            DelayExpansionLinear(self.delay_layer, 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            DelayExpansionLinear(self.delay_layer, 4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            DelayExpansionLinear(self.delay_layer, 4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg16(model_type='D', batch_norm=True):
    """
    创建VGG模型的通用函数,支持不同VGG配置。
    :param model_type: 模型类型,例如 'A', 'B', 'D', 'E'
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
    return VGG(cfg[model_type], delay_data)