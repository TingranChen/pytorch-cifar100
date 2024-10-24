"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import datetime
import utils_mine.DelayExpansion as DEX

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

    def forward(self, layer_output, batch_size, in_channels, out_channels):
        """计算每层的膨胀参数矩阵，并保存到Excel文件中。"""
        # 获取输入的形状，并判断是否需要处理
        if layer_output.dim() == 4:
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

        else:
            if batch_size != 128:
                return layer_output  # 如果batch_size不是128，直接返回
            delay_matrix = torch.zeros((batch_size, out_channels), device=layer_output.device)
            channel_mean = layer_output.mean().item()
            delay_value = self.get_closest_delay_value(channel_mean)
            delay_matrix[:, :] = delay_value

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
        base_path = "./output/average_matrix_2.xlsx"
        excel_path = base_path

        if not os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='w') as writer:
                # 创建一个占位的空 DataFrame，并写入默认子表
                pd.DataFrame().to_excel(writer, sheet_name='Placeholder')

        # 使用 'a' 模式，追加新的子表，不会覆盖已有文件
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='new') as writer:
            sheet_name = f'Run_{in_channels},{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            average_matrix_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"Average matrix of layer {in_channels}&{out_channels} saved to {excel_path} in sheet '{sheet_name}'")
        return average_matrix

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
        self.delay_layer = delay_data
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.delay_layer = DEX.DelayExpansionLayer(delay_data=self.delay_layer)

    def forward(self, x):
        in_data = x
        for layer in self.residual:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.delay_layer(x, x.size(0), layer.in_channels, layer.out_channels)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer.in_features, layer.out_features)

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
                self.delay_layer(x, x.size(0), layer.in_channels, layer.out_channels)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer.in_features, layer.out_features)

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
                self.delay_layer(x, x.size(0), layer.in_channels, layer.out_channels)
            elif isinstance(layer, nn.Linear):
                self.delay_layer(x, x.size(0), layer.in_features, layer.out_features)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        self.delay_layer(x, x.size(0), self.conv2.in_channels, self.conv2.out_channels)

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