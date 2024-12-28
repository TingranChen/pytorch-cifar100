import pandas as pd

net = 'mobile'

# 定义 Excel 文件的路径
#excel_path = "./average_matrix_vgg16.xlsx"  # 请确保路径正确
#excel_path = "./average_matrix_mobilenetv2.xlsx"  # 请确保路径正确
excel_path = "./average_matrix_inceptionv3.xlsx"  # 请确保路径正确

# 打开 Excel 文件并读取所有子表的数据
excel_data = pd.ExcelFile(excel_path)

# 初始化累加和与计数器
total_sum = 0
total_count = 0
i = 0
vgg16_repeat = [9, 9, 18, 18, 18, 36, 72, 72, 144, 288, 288, 288, 288, 32, 2, 2]
mobilenetv2_repeat = [2,2,18,1,6,54,1.5,9,91.125,1.6875,9.0,91.125,2.25,12.0,162.0,3.0,12.0,162,3,12,162,6,24,648,12,24,648,12,24,648,12,24,648,18,36,1458,27,36,1458,27,36,1458,45,75,4050,75,75,4050,75,75,4050,150,200,62.5]
inceptionv3_repeat = [18,18,36,5,108,6,4.5,100,6,54,54,27,8,6,100,8,54,54,72,9,6.75,100,9,54,54,81,486,9,54,54,72,48,56,84,48,56,56,56,84,72,72,60,87.5,105,60,87.5,87.5,87.5,105,72,72,60,87.5,105,60,87.5,105,60,87.5,87.5,87.5,105,72,72,72,126,126,72,126,126,126,126,72,72,270,72,126,126,162,200,240,216,216,280,756,216,216,120,320,384,216,216,448,756,216,216,192,1]

# 遍历所有子表，累加其中的所有有效数字并统计总数
for i, sheet_name in enumerate(excel_data.sheet_names):  
    # 读取子表数据
    sheet_data = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # 筛选出数值型数据（整数和浮点数）
    numeric_data = sheet_data.select_dtypes(include=[float, int])

    # 累加所有有效数字的和
    sheet_sum = numeric_data.sum().sum()
    #total_sum += sheet_sum * vgg16_repeat[i]
    #total_sum += sheet_sum * mobilenetv2_repeat[i]
    total_sum += sheet_sum * inceptionv3_repeat[i]
    
    # 统计所有有效数字的总数
    sheet_count = numeric_data.count().sum()
    #total_count += sheet_count*vgg16_repeat[i]
    #total_count += sheet_count*mobilenetv2_repeat[i]
    total_count += sheet_count*inceptionv3_repeat[i]

    # print(f"文件 {i} 中所有相对延迟膨胀参数累加和为：{sheet_sum},加权后为{sheet_sum*vgg16_repeat[i]}")
    # print(f"文件 {i} 中有效数字对应的总操作重复数为：{sheet_count},加权后为{sheet_count*vgg16_repeat[i]}")
    # print(f"文件 {i} 中所有元素的延迟膨胀的平均值为：{sheet_sum/sheet_count}")

    # print(f"文件 {i} 中所有相对延迟膨胀参数累加和为：{sheet_sum},加权后为{sheet_sum*mobilenetv2_repeat[i]}")
    # print(f"文件 {i} 中有效数字对应的总操作重复数为：{sheet_count},加权后为{sheet_count*mobilenetv2_repeat[i]}")
    # print(f"文件 {i} 中所有元素的延迟膨胀的平均值为：{sheet_sum/sheet_count}")

    print(f"文件 {i} 中所有相对延迟膨胀参数累加和为：{sheet_sum},加权后为{sheet_sum*inceptionv3_repeat[i]}")
    print(f"文件 {i} 中有效数字对应的总操作重复数为：{sheet_count},加权后为{sheet_count*inceptionv3_repeat[i]}")
    print(f"文件 {i} 中所有元素的延迟膨胀的平均值为：{sheet_sum/sheet_count}")

    # 记序
    i+=1
# 计算有效数字的平均值
average_value = total_sum / total_count if total_count != 0 else 0

# 打印结果
print(f"Excel 文件中所有有效数字对应的延迟膨胀累加和为：{total_sum}")
print(f"Excel 文件中有效数字对应的总操作重复数为：{total_count}")
print(f"Excel 文件中所有元素的延迟膨胀的平均值为：{average_value}")


#这是一个有若干个子表的文件。请分别累积这些子表中所有有效数据并打印出这数量等于子表数量的子表累积结果，并将这些结果累加并打印出一个总累积结果；请统计第一个子表中有效数据总数并*下方列表vgg[1]，以此类推得到各个子表独立的有效数据总数加权后的结果，打印这这些结果，随后合并这些结果得到一个总的加权后有效数据总数；
