import torch
import torch.nn as nn
import os
import pandas as pd
import datetime

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

    def forward(self, layer_output, batch_size, layer):
        """计算每层的膨胀参数矩阵，并保存到Excel文件中。"""
        # 获取输入的形状，并判断是否需要处理
        if layer_output.dim() == 4:
            batch_size, channels, height, width = layer_output.shape
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size

            if batch_size < 16:
                return layer_output  # 如果batch_size不是128，直接返回

            delay_matrix = torch.zeros((channels, height, width), device=layer_output.device)
            for c in range(channels):
                channel_mean = layer_output[:, c, :, :].mean().item()
                delay_value = self.get_closest_delay_value(channel_mean)
                delay_matrix[c, :, :] = delay_value

            # 对通道进行最大值合并，得到二维的平均矩阵
            average_matrix = delay_matrix.max(dim=0).values
            #average_matrix *= in_channels * out_channels
            #if in_channels <= 128 :
            #    average_matrix /= in_channels * 16
            #else :
            #    average_matrix /= 128 * 16

            #CIM repeat times for a single element of the mean feature
            compute_repeat = in_channels * out_channels * kernel_size[0] * kernel_size[1]
            if in_channels <= 128 :
                compute_repeat/= in_channels * 16
            else :
                compute_repeat /= 128 * 16

            print(f"Total compute Repeat is {compute_repeat} for each delay_matrix element of this layer")

        elif layer_output.dim() == 2:
            if batch_size < 16:
                return layer_output  # 如果batch_size不是128，直接返回

            in_channels = layer.in_features
            out_channels = layer.in_features

            delay_matrix = torch.zeros((batch_size, out_channels), device=layer_output.device)
            channel_mean = layer_output.mean().item()
            delay_value = self.get_closest_delay_value(channel_mean)
            delay_matrix[:, :] = delay_value

            # 对通道进行最大值合并，得到二维的平均矩阵
            average_matrix = delay_matrix.max(dim=0).values
            #average_matrix *= in_channels
            #average_matrix /= 128 * 16 # fused parallelism of CIM

            #CIM repeat times for a single element of the mean feature
            compute_repeat = in_channels
            compute_repeat /= 128 * 16 # fused parallelism of CIM

            print(f"Total compute Repeat is {compute_repeat} for each delay_matrix element of this layer")

        else:
            print(f"Not a PyTorch Layer!")

        # 执行延时膨胀计算并转换为 DataFrame
        average_matrix_df = pd.DataFrame(average_matrix.cpu().numpy())

        #确认初始路径存在
        base_dir = "./output"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)  # 如果目录不存在则创建

        # 定义初始 Excel 文件路径
        base_path = "./output/average_matrix_tmp.xlsx"
        excel_path = base_path

        if not os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, mode='w') as writer:
                # 创建一个占位的空 DataFrame，并写入默认子表
                pd.DataFrame().to_excel(writer, sheet_name='Placeholder')

        # 使用 'a' 模式，追加新的子表，不会覆盖已有文件
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='new') as writer:
            sheet_name = f'Run_{in_channels}_{out_channels},{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            average_matrix_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"Average matrix of layer {in_channels}&{out_channels} saved to {excel_path} in sheet '{sheet_name}'")
        return average_matrix
