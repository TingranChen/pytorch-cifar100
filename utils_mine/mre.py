import torch
import numpy as np


class TensorGaussianErrorWithMRE:
    def __init__(self, target_mre=0.02, std_dev=0.02):
        """
        初始化误差生成器，目标MRE为2%，标准差默认为0.02。
        :param target_mre: 目标MRE（百分比）。
        :param std_dev: 初始高斯分布误差的标准差。
        """
        self.target_mre = target_mre  # 转换为小数形式
        self.std_dev = std_dev  # 初始标准差
        self.errors = []  # 保存所有张量的相对误差
        self.device = torch.device

    def add_error(self, tensor):
        # 使用张量形式的均值和标准差，确保与输入张量的设备一致
        mean = torch.zeros(tensor.shape, device=tensor.device)
        std = torch.full(tensor.shape, self.std_dev, device=tensor.device)

        # 生成误差
        error = torch.normal(mean, std)

        # 计算误差后的张量
        tensor_with_error = tensor * (1 + error)

        # 计算相对误差并保留为张量
        relative_error = torch.abs(error).mean()  # 保持为张量
        self.errors.append(relative_error)  # 将张量添加到错误列表

        # 将错误列表的张量拼接为单个张量
        errors_tensor = torch.stack(self.errors)

        # 计算当前 MRE
        current_mre = errors_tensor.mean()  # 保持为张量

        # 打印时再转换为标量
        print(f"当前MRE: {current_mre.item():.2f}%")  # 仅在这里使用 .item()

        # 动态调整标准差
        self.adjust_std_dev(current_mre)

        return tensor_with_error

    def adjust_std_dev(self, current_mre):
        """
        根据当前MRE动态调整标准差。
        :param current_mre: 当前MRE值，类型为张量。
        """
        target_mre_tensor = torch.tensor(self.target_mre, device=current_mre.device)

        # 比较当前MRE与目标MRE，并调整标准差
        if current_mre > target_mre_tensor:
            self.std_dev *= 0.95  # 如果MRE过高，降低标准差
            print(f"标准差减小为: {self.std_dev:.6f}")
        elif current_mre < target_mre_tensor:
            self.std_dev *= 1.05  # 如果MRE过低，增加标准差
            print(f"标准差增大为: {self.std_dev:.6f}")

