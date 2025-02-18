import os
import json
import math
import pandas as pd
from collections import defaultdict


def compute_tf_idf(json_dir, word_count_file, total_docs, output_dir):
    """计算 TF-IDF 并按分块保存"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取每个文档的总词数
    word_count_df = pd.read_csv(word_count_file)
    doc_word_count = dict(zip(word_count_df["linenumber"], word_count_df["word_count"]))

    # 遍历倒排索引的所有文件
    for file_name in os.listdir(json_dir):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)

        # 结果存储字典
        tf_idf_result = defaultdict(list)

        for word, postings in inverted_index.items():
            # 文档频率 DF
            df = len(postings)

            # 计算 IDF
            idf = math.log(total_docs / (df + 1))

            for doc_id, term_freq in postings:
                # 获取该文档的总词数
                doc_word_count_val = doc_word_count.get(int(doc_id), 0)
                if doc_word_count_val == 0:
                    continue  # 跳过无效文档

                # 计算 TF
                tf = term_freq / doc_word_count_val

                # 计算 TF-IDF
                tf_idf = tf * idf

                # 添加到结果
                tf_idf_result[word].append([doc_id, tf_idf])

        # 保存当前块的 TF-IDF 结果
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(tf_idf_result, f, ensure_ascii=False, indent=4)
        print(f"保存完成: {output_file}")


# 配置路径和参数
inverted_index_dir = "./inverted_index_chunks"  # 倒排索引存储目录

word_count_csv = "word_count.csv"  # 文档词数文件
total_document_count = 150000  # 总文档数
output_tf_idf_dir = "./tf_idf_chunks"  # TF-IDF 结果存储目录

# 计算 TF-IDF
compute_tf_idf(
    inverted_index_dir, word_count_csv, total_document_count, output_tf_idf_dir
)
