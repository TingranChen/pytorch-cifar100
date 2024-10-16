import torch.nn as nn
import torch


class Quantize4bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        scale = 15.0  # 4-bit 无符号量化
        return torch.clamp(torch.round(input * scale), 0, scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantize5bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        scale = 15.5  # 5-bit 有符号量化
        return torch.clamp(torch.round(input * scale), -scale, scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, process_fn=None):
        super(QuantizedConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()
        self.process_fn = process_fn  # 实例化额外处理函数
        # 5-bit 量化权重
        self.conv.weight.data = Quantize5bit.apply(self.conv.weight.data)

    def forward(self, x):
        # 4-bit 量化输入
        x = Quantize4bit.apply(x)

        # 执行卷积操作
        x = self.conv(x)

        # 可以在这里加上其它操作,如激活和BN等
        #if self.process_fn is not None:
        #    x = self.process_fn(x)

        x = Quantize4bit.apply(x)  # 下一层的输入必须是4-bit无符号量化

        return x


class QuantizedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, process_fn, bias=False):
        """
        :param in_features: 输入特征数
        :param out_features: 输出特征数
        :param activation_fn: 选择激活函数，默认使用 ReLU
        """
        super(QuantizedLinearLayer, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.process_fn = process_fn  # 实例化额外处理函数
        # 5-bit 量化权重
        self.linear.weight.data = Quantize5bit.apply(self.linear.weight.data)

    def forward(self, x):
        # 4-bit 量化输入
        x = Quantize4bit.apply(x)

        # 执行线性层操作
        x = self.linear(x)

        # 应用传入的激活函数
        #if self.process_fn is not None:
        #    x = self.process_fn(x)

        # 4-bit 量化输出给下一层使用
        x = Quantize4bit.apply(x)

        return x

