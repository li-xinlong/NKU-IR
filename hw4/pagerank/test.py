import pandas as pd

# 文件路径
file_path = "pagerank_results.csv"

# 读取CSV文件
df = pd.read_csv(file_path)

# 打印前5行
print("前5行：")
print(df.head())

# 打印后5行
print("\n后5行：")
print(df.tail())
