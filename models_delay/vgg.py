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

class VGG(nn.Module):
    def __init__(self, features, delay_data, num_classes=100):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.delay_layer = DEX.DelayExpansionLayer(delay_data)

    def forward(self, x):
        in_channels = 3  # 初始输入通道数

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                in_channels = layer.in_channels  # 更新输入通道数
                delay_matrix = self.delay_layer(layer_output=x, batch_size=x.size(0), layer=layer)

        x = x.view(x.size(0), -1)

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                out_channels = layer.out_features
                in_channels = layer.in_features  # 更新输入通道数
                delay_matrix = self.delay_layer(x, x.size(0) ,layer=layer)

        #x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(input_channel, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channel = v
    return nn.Sequential(*layers)

# delay extension table

def vgg_model(model_type='D', batch_norm=True):
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
    return VGG(make_layers(cfg[model_type], batch_norm=batch_norm), delay_data)

