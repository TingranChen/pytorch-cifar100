import pandas as pd
import argparse

def calculate_total_quantity(txt_file, excel_file):
    """
    计算总数量。

    Args:
        txt_file (str): txt 文件的路径。
        excel_file (str): Excel 文件的路径。

    Returns:
        float: 计算得到的总数量。
    """
    # 读取 txt 文件中的数据
    # 读取 txt 文件中的数据
    with open(txt_file, 'r') as f:
        txt_data = [float(line.strip()) for line in f]

    # 读取 Excel 文件并获取每个子表的数据量
    excel_data = pd.ExcelFile(excel_file)
    sub_table_names = excel_data.sheet_names
    sub_table_data_counts = []
    for sheet_name in sub_table_names:
        df = excel_data.parse(sheet_name)
        # 正确计算子表的数据总数：行数乘以列数
        sub_table_data_counts.append(df.shape[0] * df.shape[1])

    print(f"The sheet number is {len(sub_table_data_counts)}\n")
    print(f"The repeat data number is {len(txt_data)}\n")

    # 确保 txt 文件中的数据量与 Excel 文件中的子表数量一致
    if len(txt_data) != len(sub_table_data_counts):
        raise ValueError("txt 文件中的数据量与 Excel 文件中的子表数量不一致")

    # 计算总数量
    total_quantity = 0
    for i in range(len(txt_data)):
        total_quantity += txt_data[i] * sub_table_data_counts[i]

    return total_quantity

if __name__ == '__main__':
    # 示例用法
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='densenet121', help='net type')
    args = parser.parse_args()

    model_name = args.net

    txt_file = f'compute_repeat_element_{model_name}.txt'
    excel_file = f'average_matrix_{model_name}.xlsx'  # 请替换成您的 Excel 文件名
    # txt_file = 'compute_repeat_element_resnet34.txt'
    # excel_file = 'average_matrix_resnet34.xlsx'  # 请替换成您的 Excel 文件名

    try:
        total = calculate_total_quantity(txt_file, excel_file)
        print(f"计算得到的总数量为: {total}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_file} 或 {excel_file}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")