import pandas as pd

# 文件路径
file_path = "linenumber_title_url_anchor_body.csv"

# 加载数据
data = pd.read_csv(file_path)


# 打印前 5 条记录
print("linenumber_title_url_anchor_body.csv前 5 条记录：")
print(data.head())

# 输出文件长度
print("文件长度（行数）：", len(data))
