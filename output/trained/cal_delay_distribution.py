import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_delay_distribution(excel_file, txt_file):
    # 读取 txt 文件中的数据
    with open(txt_file, 'r') as f:
        txt_data = [float(line.strip()) for line in f]

    # 使用 pandas 读取 Excel 文件
    xls = pd.ExcelFile(excel_file)
    # 获取所有子表（sheet）的名称
    sheet_names = xls.sheet_names

    # 检查 txt_data 的长度是否与 sheet_names 相同
    if len(txt_data) != len(sheet_names):
        raise ValueError("txt_data 的元素数量必须与 Excel 文件中的子表数量相同")

    # 用于存储所有子表数据的字典
    all_data = {}

    # 遍历每个子表及其对应的系数
    for i, sheet_name in enumerate(sheet_names):
        # 读取当前子表的数据
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # 获取当前子表的系数
        coefficient = txt_data[i]
        # 将当前子表的所有数据添加到字典中，并记录乘以系数后的数量
        for column in df.columns:
            for value in df[column].dropna():
                if value not in all_data:
                    all_data[value] = 0
                all_data[value] += coefficient

    # 将字典转换为列表，便于后续处理
    values = list(all_data.keys())
    counts = list(all_data.values())

    # 将数据分为 256 组
    num_bins = 256
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / num_bins

    # 计算每个数据点所属的组
    bin_indices = ((np.array(values) - min_val) / bin_width).astype(int)

    # 统计每个组的加权数量
    binned_counts = np.zeros(num_bins)
    for i, bin_index in enumerate(bin_indices):
        # 避免索引越界
        adjusted_bin_index = min(bin_index, num_bins - 1)
        binned_counts[adjusted_bin_index] += counts[i]

    # 计算每个组的中心值作为横坐标
    bin_centers = np.array([min_val + (i + 0.5) * bin_width for i in range(num_bins)])

    # 计算概率密度
    total_count = sum(binned_counts)
    probabilities = binned_counts / total_count

    # 绘制概率分布曲线
    plt.bar(bin_centers, probabilities, width=bin_width, alpha=0.7, color='skyblue', edgecolor='black')
    # 添加标签和标题
    # plt.xlabel('元素值区间中位值')
    # plt.ylabel('概率密度')
    # plt.title('所有子表中元素乘以系数后的概率分布 (256 组)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(right=1.3)
    plt.xlim(left=-0.1)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.show()
    # 关闭 Excel 文件
    xls.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenetv2', help='net type')
    args = parser.parse_args()
    excel_file = f"average_matrix_{args.net}.xlsx"
    txt_file = f"compute_repeat_element_{args.net}.txt"
    plot_delay_distribution(excel_file, txt_file)