import pandas as pd

# 定义 Excel 文件的路径
excel_path = "./average_matrix_vgg.xlsx"  # 请确保路径正确

# 打开 Excel 文件并读取所有子表的数据
excel_data = pd.ExcelFile(excel_path)

# 初始化累加和与计数器
total_sum = 0
total_count = 0

# 遍历所有子表，累加其中的所有有效数字并统计总数
for sheet_name in excel_data.sheet_names:
    # 读取子表数据
    sheet_data = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # 筛选出数值型数据（整数和浮点数）
    numeric_data = sheet_data.select_dtypes(include=[float, int])
    
    # 累加所有有效数字的和
    total_sum += numeric_data.sum().sum()
    
    # 统计所有有效数字的总数
    total_count += numeric_data.count().sum()

# 计算有效数字的平均值
average_value = total_sum / total_count if total_count != 0 else 0

# 打印结果
print(f"Excel 文件中所有有效数字的累加和为：{total_sum}")
print(f"Excel 文件中有效数字的总数为：{total_count}")
print(f"Excel 文件中有效数字的平均值为：{average_value}")