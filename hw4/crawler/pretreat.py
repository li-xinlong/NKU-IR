import pandas as pd

# 文件路径
file_path = "title_url_anchor_body.csv"

# 加载数据
data = pd.read_csv(file_path)

# 在第一列前插入行号
data.insert(0, "line_number", range(1, len(data) + 1))

# 打印前 5 条记录
print("前 5 条记录：")
print(data.head())

# 打印后 5 条记录
print("\n后 5 条记录：")
print(data.tail())

# 保存处理后的文件
output_path = "linenumber_title_url_anchor_body.csv"
data.to_csv(output_path, index=False)

print(f"\n处理完成，已保存到 {output_path}")
