import torch
import torch.nn as nn

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # 卷积层 (32, 32, 128, 128) -> (32, 32, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 平均池化层 (32, 32, 128, 128) -> (32, 32, 64, 64)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 激活函数 (ReLU)
        self.relu = nn.ReLU()

        # 批归一化层
        self.bn = nn.BatchNorm2d(32)

        # 全连接层 (32 * 64 * 64 = 131072) -> (16)
        # 根据描述全连接层是(16,162,18,16), 这个理解可能有误
        # 这里按逻辑推理, 应该是(32,64,64) flatten之后是131072, 然后连接到16个神经元
        self.fc1 = nn.Linear(32 * 64 * 64, 16)

    def forward(self, x):
        # 卷积层
        x = self.conv1(x)

        # 平均池化层
        x = self.pool(x)

        # 激活函数 + 批归一化层
        x = self.relu(x)
        x = self.bn(x)

        # 将特征图展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)

        return x

# 创建一个随机的输入特征图 (32, 32, 128, 128)
input_tensor = torch.randn(32, 32, 128, 128)

# 创建模型实例
model = MyModel()

# 将输入传入模型进行前向传播
output = model(input_tensor)

# 打印输出结果的尺寸
print("Output shape:", output.shape)  # 预期输出: torch.Size([32, 16])

# 打印模型结构
print(model)