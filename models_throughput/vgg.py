import torch
import torch.nn as nn
import pandas as pd
import datetime
import os

# VGG 的配置参数
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class DelayExpansionLayer(nn.Module):
    def __init__(self, delay_data):
        super().__init__()
        self.delay_map = self._create_delay_map(delay_data)

    def _create_delay_map(self, delay_data):
        """将延时膨胀数据转换为映射表。"""
        return {round(row['data'], 6): row['delay expension'] for _, row in delay_data.iterrows()}

    def get_closest_delay_value(self, mean_value):
        """查找距离最近的延时膨胀参数。"""
        rounded_mean = round(mean_value, 6)

        if rounded_mean in self.delay_map:
            return self.delay_map[rounded_mean]

        closest_key = min(self.delay_map.keys(), key=lambda k: abs(k - rounded_mean))
        return self.delay_map[closest_key]

    def forward(self, layer_output, in_channels, out_channels):
        """计算每层的膨胀参数矩阵，并保存到Excel文件中。"""
        batch_size, channels, height, width = layer_output.shape
        if batch_size != 128:
            return layer_output  # 如果batch_size不是128，直接返回

        delay_matrix = torch.zeros((channels, height, width), device=layer_output.device)
        for c in range(channels):
            channel_mean = layer_output[:, c, :, :].mean().item()
            delay_value = self.get_closest_delay_value(channel_mean)
            delay_matrix[c, :, :] = delay_value

        # 对通道进行最大值合并，得到二维的平均矩阵
        average_matrix = delay_matrix.max(dim=0).values
        average_matrix *= in_channels * out_channels
        average_matrix /= (in_channels * 16 if in_channels <= 128 else 128 * 16)

        # 执行延时膨胀计算并转换为 DataFrame
        average_matrix_df = pd.DataFrame(average_matrix.cpu().numpy())

        #确认初始路径存在
        base_dir = "./output"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)  # 如果目录不存在则创建

        # 定义初始 Excel 文件路径
        base_path = "./output/average_matrix_1.xlsx"
        excel_path = base_path

        if not os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='w') as writer:
                # 创建一个占位的空 DataFrame，并写入默认子表
                pd.DataFrame().to_excel(writer, sheet_name='Placeholder')

        # 使用 'a' 模式，追加新的子表，不会覆盖已有文件
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='new') as writer:
            sheet_name = f'Run_{in_channels},{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            average_matrix_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"Average matrix of layer {in_channels} saved to {excel_path} in sheet '{sheet_name}'")
        return average_matrix

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
        self.delay_layer = DelayExpansionLayer(delay_data)

    def forward(self, x):
        all_layers_delay_matrices = []
        in_channels = 3  # 初始输入通道数

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                in_channels = layer.in_channels  # 更新输入通道数
                delay_matrix = self.delay_layer(x, in_channels, out_channels)
                all_layers_delay_matrices.append(delay_matrix)

        x = x.view(x.size(0), -1)

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                in_channels = layer.in_channels  # 更新输入通道数
                delay_matrix = self.delay_layer(x, in_channels, out_channels)
                all_layers_delay_matrices.append(delay_matrix)

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

