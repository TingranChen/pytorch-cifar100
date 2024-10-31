import torch
import numpy as np
from collections import defaultdict

def load_model_weights(pth_file):
    """加载 .pth 文件中的模型权重"""
    model_weights = torch.load(pth_file, map_location='cpu')
    # 如果保存的是state_dict，需要取出state_dict
    if 'state_dict' in model_weights:
        model_weights = model_weights['state_dict']
    return model_weights

def process_weights(weights):
    """将权重值保留到小数点后2位，并统计其数量"""
    rounded_weights = np.round(weights.numpy(), 2)
    unique, counts = np.unique(rounded_weights, return_counts=True)
    return dict(zip(unique, counts))

def analyze_weights(pth_file):
    """分析卷积层和全连接层的权重信息"""
    model_weights = load_model_weights(pth_file)

    # 用于统计权重总数和分布
    total_weight_count = 0
    weight_distribution = defaultdict(int)

    for name, param in model_weights.items():
        if 'weight' in name:  # 只处理权重，忽略bias等
            # 检查是否为卷积或全连接层的权重
            if len(param.shape) in [2, 4]:  # Linear: 2D, Conv: 4D
                # 统计当前层的权重
                layer_distribution = process_weights(param)
                layer_total = param.numel()

                # 更新总体统计
                total_weight_count += layer_total
                for value, count in layer_distribution.items():
                    weight_distribution[value] += count

    return total_weight_count, dict(weight_distribution)

def print_weight_statistics(total_weight_count, weight_distribution):
    """打印权重统计信息"""
    print(f"Total number of weights: {total_weight_count}")
    print("Weight Distribution:")
    for value, count in sorted(weight_distribution.items()):
        print(f"Value: {value:.2f}, Count: {count}")

# 示例：使用模型.pth文件路径
#pth_file = "./checkpoint/vgg16/quan/Wednesday_30_October_2024_11h_46m_00s/vgg16-1-regular.pth"
#pth_file = "./checkpoint/mobilenetv2/quan/Wednesday_30_October_2024_11h_15m_55s/mobilenetv2-1-regular.pth"
#pth_file = "./checkpoint/inceptionv3/quan/Saturday_26_October_2024_10h_12m_28s/inceptionv3-1-regular.pth"

#pth_file = "./checkpoint/mobilenetv2/"
pth_file = "./checkpoint/vgg16/Thursday_31_October_2024_14h_35m_28s/vgg16-1-regular.pth"

total_count, distribution = analyze_weights(pth_file)
print_weight_statistics(total_count, distribution)
