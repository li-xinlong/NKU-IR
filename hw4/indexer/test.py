def count_lines(file_path):
    """统计文件的行数"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            line_count = sum(1 for _ in file)
        return line_count
    except FileNotFoundError:
        return "文件不存在"
    except Exception as e:
        return f"发生错误: {e}"


# 文件路径
file_path = "file_word_count.csv"

# 调用函数统计行数
line_count = count_lines(file_path)
print(f"文件行数: {line_count}")
