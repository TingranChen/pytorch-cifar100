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

    def max_in_groups_average(self, delay_matrix, group_size=16):
        """
           Processes a delay matrix, which can be either 2D (fully connected) or 3D (convolutional).

           For 3D matrices:
               1. Calculates the maximum within groups of consecutive channels.
               2. Averages these maximum values across the groups.

           For 2D matrices:
               1. Calculates the maximum within groups of consecutive elements.
               2. Averages these maximum values across the groups.

           Args:
               delay_matrix: A PyTorch tensor representing the delay matrix (2D or 3D).
               group_size: The size of each group.

           Returns:
               A PyTorch tensor of shape (H, W) for 3D input or (D) for 2D input,
               containing the averaged maximum delay values.
           """
        if delay_matrix.dim() == 3:
            channels, height, width = delay_matrix.shape
            delay_matrix_max = torch.zeros(channels, height, width)

            # 处理每group_size个通道
            for i in range((channels + group_size-1) // group_size):
                start_idx = i * group_size
                end_idx = min((i + 1) * group_size, channels)  # 最后一组可能不足group_size个通道
                # max_values 的形状将会是 (height, width)
                max_values = delay_matrix[start_idx:end_idx, :, :].max(dim=0).values # 计算每组的最大值 (在通道维度上取最大值)
                # 将最大值赋给 delay_matrix 的这group_size个通道
                delay_matrix_max[start_idx:end_idx, :, :] = max_values # 利用广播机制，max_values (height, width) 会被自动扩展为 (end_idx-start_idx, height, width)

            average_matrix = delay_matrix_max.mean(dim=0) # 在通道维度上取平均
            return average_matrix


        elif delay_matrix.dim() == 2:
            # Fully Connected Layer Output (2D)
            batch, out_channels = delay_matrix.shape
            average_matrix = torch.zeros(out_channels, device=delay_matrix.device)
            for i in range((out_channels + group_size-1) // group_size):
                start_idx = i * group_size
                end_idx = min((i + 1) * group_size, out_channels)  # 处理最后一组可能不足group_size列的情况
                # 计算每组的最大值 (沿着行维度计算最大值)
                # 由于每列相同，这里取第0行的最大值即可
                max_value = delay_matrix[0, start_idx:end_idx].max()
                average_matrix[start_idx:end_idx] = max_value

            return average_matrix
        else:
            raise ValueError("Delay matrix must be either 2D or 3D.")

    def forward(self, layer_output, batch, in_channels, out_channels, kernel_size):
        """计算每层的膨胀参数矩阵，并保存到Excel文件中。"""
        # 获取输入的形状，并判断是否需要处理
        if layer_output.dim() == 4:
            type = "CONV"
            batch_size, channels, height, width = layer_output.shape #输出特征图尺寸信息
            assert channels == out_channels #输出特征图的通道数必然等于算法层的输出通道数

            if batch_size < 16:
                return layer_output  # 如果batch_size过小，直接返回

            delay_matrix = torch.zeros((channels, height, width), device=layer_output.device)
            for c in range(channels):
                channel_mean = layer_output[:, c, :, :].mean().item()
                delay_value = self.get_closest_delay_value(channel_mean)
                delay_matrix[c, :, :] = delay_value

            # delay_matrix = layer_output.mean(dim=0)

            average_matrix = self.max_in_groups_average(delay_matrix, 16) #每16个相邻通道取最大值

            #CIM repeat times for a single element of the mean feature
            compute_repeat = in_channels * out_channels * kernel_size[0] * kernel_size[1]
            if in_channels <= 128 : #卷积输入通道数小于CIM的计算输入并行度
                compute_repeat /= in_channels
            else :
                compute_repeat /= 128

            if out_channels <= 16 : #卷积输入通道数小于CIM的计算输出并行度
                compute_repeat /= out_channels
            else :
                compute_repeat /= 16

            #print(f"Total compute Repeat is {compute_repeat} for each delay_matrix element of this layer")
            print(f"{compute_repeat}")

        elif layer_output.dim() == 2:
            batch_size, channels = layer_output.shape
            type = "FC"

            if batch_size < 16:
                return layer_output  # 如果batch_size不是128，直接返回

            in_channels = layer.in_features
            out_channels = layer.out_features

            # 沿着第一个维度 (batch_size) 计算平均值，得到一个形状为 (out_channels,) 的一维张量
            channel_means = layer_output.mean(dim=0)
            # 为每个 channel 获取对应的 delay_value
            delay_values = []
            for channel_mean in channel_means:
                delay_value = self.get_closest_delay_value(channel_mean.item())
                delay_values.append(delay_value)

            # 将 delay_values 转换为张量
            delay_values = torch.tensor(delay_values, device=layer_output.device)
            # 将 delay_values 赋值给 delay_matrix 的每一行
            delay_matrix = delay_values.repeat(batch_size, 1)

            # 对通道进行最大值合并，得到二维的平均矩阵
            average_matrix = self.max_in_groups_average(delay_matrix, 16) #每16个相邻通道取最大值

            #CIM repeat times for a single element of the mean feature
            compute_repeat = in_channels
            if in_channels <= 128 : #卷积输入通道数小于CIM的计算输入并行度
                compute_repeat /= in_channels
            else :
                compute_repeat /= 128

            if out_channels <= 16 : #卷积输入通道数小于CIM的计算输出并行度
                compute_repeat /= out_channels
            else :
                compute_repeat /= 16

            #print(f"Total compute Repeat is {compute_repeat} for each delay_matrix element of this layer")
            print(f"{compute_repeat}")

        else:
            print(f"Not a PyTorch Layer!")

        # 执行延时膨胀计算并转换为 DataFrame
        average_matrix_df = pd.DataFrame(average_matrix.detach().cpu().numpy())

        # 确认初始路径存在
        base_dir = "./output/pure_Throughput"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 定义 Excel 文件路径
        excel_path = os.path.join(base_dir, "average_matrix_tmp.xlsx")

        # 使用 'w' 模式，覆写整个Excel文件
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            sheet_name = f'{type}_{in_channels}_{out_channels},{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            average_matrix_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        #print(f"Average matrix of layer {in_channels}&{out_channels} saved to {excel_path} in sheet '{sheet_name}'")
        return average_matrix