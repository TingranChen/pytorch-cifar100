import torch
import torch.nn as nn
import os
import pandas as pd
import datetime
import math

class DelayCalculationLayer(nn.Module):
   def __init__(self):
        super().__init__()
        self.row = 1 # 多核架构的行数
        self.col = 16 # 多核架构的列数
        self.cinUnit = 128 # 一个Core的MAC计算的输入通道
        self.coutUnit = 16 # 一个Core的MAC计算的输出通道
        self.latency = 10 # core的计算延时 (CLK)
        self.bandwidth = 512 # 输入带宽 (bit/CLK)
        self.precision = 4 # 数据精度(bit)
        self.paral_pix = 256 # 像素并行度（用于容纳乒乓权重更新带来的事件开销）
        self.clk_period = 2 # 时钟周期 (ns)
        self.paralSIMD = 16 # SIMD电路的计算并行度 

   def find_optimal_rectangle_dimensions(self, length_unit, width_unit, core_number, input_channel, output_channel):
       """
       找到最佳的长方形边长，使得用最少的长方形来填充一个指定区域。

       Args:
           length_unit (int): 长方形长度的单位。
           width_unit (int): 长方形宽度的单位。
           core_number (int): 用于计算总面积的核数。
           input_channel (int): 目标区域的长度。
           output_channel (int): 目标区域的宽度。

       Returns:
           tuple: 包含最佳长方形长度、宽度和所需数量的元组 (length, width, count)。
                  如果找不到合适的边长，返回 None。
       """
       # 计算总面积
       total_area = length_unit * width_unit * core_number

       # 初始化最佳解
       best_length = None
       best_width = None
       min_count = float('inf')

       # 遍历所有可能的 k 和 m
       for k in range(1, core_number + 1):
           if core_number % k == 0:
               for m in range(1,math.ceil(core_number / k) + 1):
                   length = k * length_unit
                   width = m * width_unit

                   # # 检查面积是否匹配
                   # if length * width <= total_area:
                   #     continue

                   # 计算所需长方形数量
                   num_length = math.ceil(input_channel / length)
                   num_width = math.ceil(output_channel / width)
                   count = num_length * num_width

                   # 更新最佳解
                   if count < min_count:
                       min_count = count
                       best_length = k
                       best_width = m

       # 返回结果
       if best_length is not None and best_width is not None:
           return best_length, best_width, min_count
       else:
           return None

   def forward(self, layer_output, batch, layer):
        """计算每层的膨胀参数矩阵，并保存到Excel文件中。"""
        # 获取输入的形状，并判断是否需要处理
        if layer_output.dim() == 4:
            type = "CONV"
            batch_size, channels, height, width = layer_output.shape #输出特征图尺寸信息
            in_channels = layer.in_channels #layer 是算法层
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            assert channels == in_channels #输入特征图的通道数必然等于算法层的输出通道数

            #计算fixed rowxcol情况下的计算延时
            #总共有多少次MAC计算
            compute_repeat = in_channels * out_channels * kernel_size[0] * kernel_size[1] * height * width / stride[0] / stride[1]
            if in_channels <= self.cinUnit * self.row : #卷积输入通道数小于CIM阵列的计算输入并行度
                compute_repeat /= in_channels
                row_size = math.ceil(in_channels/self.cinUnit)
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_in = math.ceil(in_channels / (self.cinUnit * self.row))
                equal_cin = math.ceil(in_channels/ repeat_times_in)#计算归一化计算并行度
                compute_repeat /= equal_cin
                row_size = self.row

            if out_channels <= self.coutUnit * self.col : #卷积输出通道数小于CIM阵列的计算输出并行度
                compute_repeat /= out_channels
                col_size = math.ceil(out_channels/self.coutUnit)
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_out = math.ceil(out_channels / (self.coutUnit * self.col))
                equal_cout = math.ceil(out_channels/ repeat_times_out)#计算归一化计算并行度
                compute_repeat /= equal_cout
                col_size = self.col

            # 每次数据输入耗费的时间（CLK）
            data_transfer_in_time_unit = row_size * self.cinUnit * self.precision / self.bandwidth
            # 每次结果输出耗费的时间（CLK）
            data_transfer_out_time_unit = col_size * self.coutUnit * self.precision / self.bandwidth
            # 每次MAC计算耗费的时间（CLK）
            mac_time_unit = self.latency

            # 数据输入总的时间
            data_transfer_in_time_fixed = data_transfer_in_time_unit * compute_repeat * self.clk_period
            # 结果输出总的时间
            data_transfer_out_time_fixed = data_transfer_out_time_unit * compute_repeat * self.clk_period
            # 计算MAC计算总的时间
            mac_time_fixed = mac_time_unit * compute_repeat * self.clk_period

            #计算权重更新的时间开销
            weight_update_time_cores = row_size * col_size * self.cinUnit * self.coutUnit * self.precision / self.bandwidth #当前多核结构内全部权重数据更新花费的CLK数
            weight_updata_repeat_unit = math.ceil(in_channels * out_channels * kernel_size[0] * kernel_size[1] / row_size / col_size /self.cinUnit / self.coutUnit) #要完成paral_pix个像素的卷积，多核架构上权重的刷新次数（全局的）
            mac_repeat_per_weight_update = math.ceil(weight_update_time_cores / self.latency) # 要能够实现乒乓，要在一次权重刷新中实现的mac运算的最少次数
            assert self.paral_pix > mac_repeat_per_weight_update # 设定的像素并行度参数必须大于由其它参数计算得到的最少次数,否则无法实现乒乓
            if self.paral_pix > height*width :
                paral_pix = height*width
            else:
                paral_pix = self.paral_pix
            weight_time_fixed = math.ceil(height * width / paral_pix * weight_updata_repeat_unit * weight_update_time_cores * self.clk_period) #整个卷积层上执行全部计算所需的权重写入时间

            #计算dynamic rowxcol情况下的计算延时(优先满足输入并行度)
            #总共有多少次MAC计算
            compute_repeat = in_channels * out_channels * kernel_size[0] * kernel_size[1] * height * width
            row_size, col_size, number = self.find_optimal_rectangle_dimensions(self.cinUnit, self.coutUnit, self.row * self.col, in_channels, out_channels)
            # assert (row_size * col_size == self.row* self.col)

            if in_channels <= self.cinUnit * row_size : #卷积输入通道数小于CIM阵列的计算输入并行度
                compute_repeat /= in_channels
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_in = math.ceil(in_channels / (self.cinUnit * row_size))
                equal_cin = math.ceil(in_channels / repeat_times_in) #计算归一化计算并行度
                compute_repeat /= equal_cin

            if out_channels <= self.coutUnit * self.col : #卷积输出通道数小于CIM阵列的计算输出并行度
                compute_repeat /= out_channels
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_out = math.ceil(out_channels / (self.coutUnit * col_size))
                equal_cout = math.ceil(out_channels/ repeat_times_out)  #计算归一化计算并行度
                compute_repeat /= equal_cout

            repeat_times_pix = math.ceil(self.row * self.col / (row_size*col_size))
            if repeat_times_pix >= kernel_size[0]*kernel_size[1] : #可用的像素并行度超过卷积窗口大小
                repeat_times_pix = kernel_size[0]*kernel_size[1]
                compute_repeat /= repeat_times_pix
            else :
                # equal_pix = kernel_size[0]*kernel_size[1] / repeat_times_pix  #计算归一化计算并行度
                compute_repeat /= repeat_times_pix

            # 每次数据输入耗费的时间（CLK）
            data_transfer_in_time_unit = math.ceil(repeat_times_pix * row_size * self.cinUnit * self.precision / self.bandwidth)
            # 每次结果输出耗费的时间（CLK）
            data_transfer_out_time_unit = math.ceil(col_size * self.coutUnit * self.precision / self.bandwidth)
            # 每次MAC计算耗费的时间（CLK）
            mac_time_unit = self.latency

            # 数据输入总的时间
            data_transfer_in_time_dynamic = data_transfer_in_time_unit * compute_repeat * self.clk_period
            # 结果输出总的时间
            data_transfer_out_time_dynamic = data_transfer_out_time_unit * compute_repeat * self.clk_period
            # 计算MAC计算总的时间
            mac_time_dynamic = mac_time_unit * compute_repeat * self.clk_period

            #计算权重更新总的时间
            weight_update_time_cores = row_size * col_size * repeat_times_pix * self.cinUnit * self.coutUnit * self.precision / self.bandwidth #当前多核结构内全部权重数据更新花费的CLK数
            weight_updata_repeat_unit = math.ceil(in_channels * out_channels * kernel_size[0] * kernel_size[1] / repeat_times_pix / row_size / col_size /self.cinUnit / self.coutUnit) #要完成paral_pix个像素的卷积，多核架构上权重的刷新次数（全局的）
            mac_repeat_per_weight_update = math.ceil(weight_update_time_cores / self.latency) # 要能够实现乒乓，要在一次权重刷新中实现的mac运算的最少次数
            assert self.paral_pix > mac_repeat_per_weight_update # 设定的像素并行度参数必须大于由其它参数计算得到的最少次数,否则无法实现乒乓
            if self.paral_pix > height*width :
                paral_pix = height*width
            else:
                paral_pix = self.paral_pix
            weight_time_dynamic = math.ceil(height * width / paral_pix * weight_updata_repeat_unit * weight_update_time_cores * self.clk_period) #整个卷积层上执行全部计算所需的权重写入时间

            print(f"{height} {width} {in_channels} {out_channels} {repeat_times_pix} {max(mac_time_unit,data_transfer_in_time_unit,data_transfer_out_time_unit)} {row_size} {col_size} {data_transfer_in_time_fixed} {mac_time_fixed} {data_transfer_out_time_fixed} {weight_time_fixed} {data_transfer_in_time_dynamic} {mac_time_dynamic} {data_transfer_out_time_dynamic} {weight_time_dynamic}")

        elif layer_output.dim() == 2:
            batch_size, channels = layer_output.shape
            type = "FC"

            in_channels = layer.in_features
            out_channels = layer.out_features

            #计算fixed rowxcol情况下的计算延时\
            #总共有多少次MAC计算
            compute_repeat = in_channels * out_channels
            if in_channels <= self.cinUnit * self.row : #卷积输入通道数小于CIM阵列的计算输入并行度
                compute_repeat /= in_channels
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_in = math.ceil(in_channels / self.cinUnit * self.row)
                equal_cin = math.ceil(in_channels/ repeat_times_in)#计算归一化计算并行度
                compute_repeat /= equal_cin * self.row

            if out_channels <= self.coutUnit * self.col : #卷积输出通道数小于CIM阵列的计算输出并行度
                compute_repeat /= out_channels
            else :  #卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_out = math.ceil(out_channels / self.coutUnit * self.col)
                equal_cout = math.ceil(out_channels/ repeat_times_out)#计算归一化计算并行度
                compute_repeat /= equal_cout * self.col

            # 计算MAC计算总的时间
            mac_time_fixed = self.latency * compute_repeat * self.clk_period
            # 计算数据传输总的时间
            datatransfer_time = self.row * self.cinUnit * self.precision / self.bandwidth * compute_repeat * self.latency


            # 计算dynamic rowxcol情况下的计算延时(优先满足输入并行度)
            # 总共有多少次MAC计算
            compute_repeat = in_channels * out_channels
            row_size, col_size, number = self.find_optimal_rectangle_dimensions(self.cinUnit, self.coutUnit, self.row * self.col, in_channels, out_channels)

            if in_channels <= self.cinUnit * row_size:  # 卷积输入通道数小于CIM阵列的计算输入并行度
                compute_repeat /= in_channels
            else:  # 卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_in = math.ceil(in_channels / (self.cinUnit * row_size))
                equal_cin = math.ceil(in_channels / repeat_times_in)  # 计算归一化计算并行度
                compute_repeat /= equal_cin * row_size

            if out_channels <= self.coutUnit * self.col:  # 卷积输出通道数小于CIM阵列的计算输出并行度
                compute_repeat /= out_channels
            else:  # 卷积输入通道数大于CIM阵列的计算输入并行度
                repeat_times_out = math.ceil(out_channels / (self.coutUnit * col_size))
                equal_cout = math.ceil(out_channels / repeat_times_out)  # 计算归一化计算并行度
                compute_repeat /= equal_cout * col_size

            # 每次MAC计算耗费的时间（CLK）
            datatransfer_time = row_size * self.cinUnit * self.precision / self.bandwidth
            mac_time = max(datatransfer_time, self.latency)
            # 计算MAC计算总的时间
            mac_time_dynamic = mac_time * compute_repeat * self.clk_period
            # print(f"Total compute Repeat is {compute_repeat} for each delay_matrix element of this layer")

            print(f"- - {in_channels} {out_channels} - {mac_time} {row_size} {col_size} 0 {mac_time_fixed} 0 2048 0 {mac_time_dynamic} 0 5096")

        else:
            print(f"Not a PyTorch Layer!")

        # 执行延时膨胀计算并转换为 DataFrame

        # 确认初始路径存在
        if batch_size == 1:
            base_dir = "./output/trained/single_b"
        else:
            base_dir = "./output/trained"

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)  # 如果目录不存在则创建

        # 定义 Excel 文件路径
        excel_path = os.path.join(base_dir, "single_matrix_tmp.xlsx")

        # 使用 'a' 模式，追加新的子表，不会覆盖已有文件
        # with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='new') as writer:
        #     sheet_name = f'{type}_{in_channels}_{out_channels},{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        #     average_matrix_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        #
        # #print(f"Average matrix of layer {in_channels}&{out_channels} saved to {excel_path} in sheet '{sheet_name}'")
        # return average_matrix

   def rank_transform(self, input_matrix, output_matrix, has_rank_transform):
       """
       计算维度转秩的延时数据，适用于 Transformer 网络。

       Args:
           input_matrix (torch.Tensor): 输入矩阵，二维，形状为 (序列长度, 特征维度)。
           output_matrix (torch.Tensor): 输出矩阵，二维，形状为 (序列长度, 特征维度)。
           has_rank_transform (bool): 是否有转秩电路。

       Returns:
           tuple: 包含延时数据的元组 (read_delay, write_delay)。
       """
       seq_length, input_features = input_matrix.size()
       _, output_features = output_matrix.size()

       # 计算输入的读时钟数
       if input_features*self.precision < self.bandwidth: # 一个clk可以读完一个
           read_delay = seq_length * self.clk_period  # 计算所需的时钟周期
       else:
           read_delay = (seq_length * input_features) // self.bandwidth * self.clk_period  # 计算所需的时钟周期

       # 计算输出的写时钟数
       if input_features*self.precision < self.bandwidth: # 一个clk可以读完一个
           read_delay = seq_length * self.clk_period  # 计算所需的时钟周期
       else:
           read_delay = (seq_length * input_features) // self.bandwidth * self.clk_period  # 计算所需的时钟周期

       # 计算输出的写时钟数
       if output_features*self.precision < self.bandwidth: # 一个clk可以读完一个
           write_delay = seq_length * self.clk_period  # 计算所需的时钟周期
       else:
           write_delay = (seq_length * output_features) // self.bandwidth * self.clk_period  # 计算所需的时钟周期


       # 如果有转秩电路，可能需要额外的延时
       if has_rank_transform:
           read_delay = int(read_delay / self.coutUnit)  # 假设转秩电路增加50%的延时
           write_delay = int(write_delay / self.cinUnit)  # 假设转秩电路增加50%的延时

       print(f"{input_features} {output_features} {read_delay} {write_delay}")
       
       return read_delay, write_delay
    
   def pooling_branch(self, pooling_layer, input_matrix, output_matrix, has_topk):
        """
        计算池化分支的延时数据。
        """
        batch_size, input_channels, input_height, input_width = input_matrix.size()
        _, output_channels, output_height, output_width = output_matrix.size()

        # 计算池化运算的运算总量
        if isinstance(pooling_layer, nn.MaxPool2d) or isinstance(pooling_layer, nn.AvgPool2d):
            kernel_size = pooling_layer.kernel_size
            stride = pooling_layer.stride if pooling_layer.stride is not None else kernel_size
            output_height = (input_height - kernel_size) // stride + 1
            output_width = (input_width - kernel_size) // stride + 1
            total_operations = output_channels * output_height * output_width * (kernel_size ** 2)  # 每个输出位置的运算量

        elif isinstance(pooling_layer, nn.AdaptiveAvgPool2d):
            # 对于 AdaptiveAvgPool2d，输出大小通常为 (1, 1)
            output_height, output_width = output_matrix.size()[2:]
            # 假设每个输出位置需要计算整个输入区域的平均值
            kernel_size = (input_height // output_height, input_width // output_width)
            total_operations = output_channels * output_height * output_width * (kernel_size[0] * kernel_size[1])  # 每个输出位置的运算量

        else:
            raise ValueError("Unsupported pooling layer type.")

        # 计算延时
        parallelism  = self.paralSIMD
        total_delay = total_operations // parallelism  # 计算所需的时钟周期
        if total_operations % parallelism != 0:
            total_delay += 1  # 如果有余数，增加一个时钟周期

        # 根据是否有 TOPk 操作调整延时
        if has_topk and isinstance(pooling_layer, nn.MaxPool2d):
            topk_delay = total_delay // min(self.cinUnit, input_channels) // min(kernel_size, output_channels) * self.clk_period  # 根据新逻辑计算 TOPk 延时
        else:
            topk_delay = total_delay * self.clk_period  # 非 TOPk 的延时与总延时相同

        non_topk_delay = total_delay * self.clk_period  # 非 TOPk 的延时与总延时相同

        print(f"{input_channels} {output_channels} {topk_delay} {non_topk_delay}")

        return topk_delay, non_topk_delay

   def activation_branch(self, activation_layer, input_matrix):
        """
        计算激活函数分支的延时数据。

        Args:
            activation_layer (nn.Module): 激活函数层（ReLU 或 Softmax）。
            input_matrix (torch.Tensor): 输入矩阵（FC二维或CNN三维）。
            output_matrix (torch.Tensor): 输出矩阵（FC二维或CNN三维）。
            parallelism (int): 计算并行度。

        Returns:
            tuple: 包含延时数据的元组 (relu_delay, softmax_delay)。
                  其中 softmax_delay 包含两种情况：全部计算分支或TOPk分支的延时。
        """
        # 获取输入矩阵的形状和计算数据总量
        if input_matrix.dim() == 2:  # FC 层，二维
            batch_size, features = input_matrix.size()
            total_elements = features
        elif input_matrix.dim() == 4:  # CNN 层，四维
            batch_size, channels, height, width = input_matrix.size()
            total_elements = channels * height * width
        else:
            raise ValueError("Unsupported input matrix dimension")

        # 计算SIMD的并行度
        parallelism = self.paralSIMD

        # 计算ReLU激活函数运算的延时
        if isinstance(activation_layer, nn.ReLU):
            relu_latency = 1  # ReLU计算latency (CLK)
            relu_delay = (total_elements // parallelism) * relu_latency * self.clk_period
            if total_elements % parallelism != 0:
                relu_delay += relu_latency * self.clk_period
        else:
            relu_delay = 0

        full_softmax_delay = 0
        topk_softmax_delay = 0
        # 计算Softmax激活函数运算的延时
        if isinstance(activation_layer, nn.Softmax):
            softmax_latency = 5  # Softmax计算latency (CLK)
            # 全部计算分支
            full_softmax_delay = (total_elements // parallelism) * softmax_latency * self.clk_period
            if total_elements % parallelism != 0:
                full_softmax_delay += softmax_latency * self.clk_period
            
            # TOPk分支（假设k=5）
            k = 5
            if k >= parallelism:
                topk_softmax_delay = math.ceil(k // parallelism) * softmax_latency * self.clk_period
            else:
                topk_softmax_delay += (k * total_elements) % parallelism * self.clk_period

        print(f"{input_matrix.size()} {relu_delay} {full_softmax_delay} {topk_softmax_delay}")