import os
import json
import pypinyin  # 用于生成拼音


# 获取词对应的拼音形式
def generate_pinyin(term):
    """生成拼音形式"""
    pinyin_list = pypinyin.lazy_pinyin(term)
    return "".join(pinyin_list)


# 获取英文形式（如直接处理拼音或简单映射，当前假设词语无复杂映射）
def generate_english_associations(term):
    """生成简单的英文形式（可扩展逻辑）"""
    return [term.lower()]


# 获取词对应的TF-IDF文件名称
def get_tf_idf_file_for_term(term):
    """根据词汇获取对应TF-IDF文件名"""
    if "\u4e00" <= term[0] <= "\u9fff":
        return f"chinese_{term[0]}.json"
    elif term[0].isalpha():
        return f"alpha_{term[0].lower()}.json"
    elif term[0].isdigit():
        return "numeric.json"
    else:
        return "others.json"


# 查询联想词是否存在于TF-IDF文件中
def search_associated_terms(term, tf_idf_dir):
    """查询拼音和英文形式的词是否存在（支持包含关系，中文优先返回）"""
    associated_terms = set()
    pinyin_form = generate_pinyin(term)
    english_forms = generate_english_associations(term)

    # 查询拼音和英文形式的词是否存在于TF-IDF中
    all_associations = [pinyin_form] + english_forms

    for associated_term in all_associations:
        file_name = get_tf_idf_file_for_term(associated_term)
        file_path = os.path.join(tf_idf_dir, file_name)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 查找文件中所有包含查询词的词语
                for word in data.keys():
                    if associated_term in word:  # 判断是否包含查询词
                        associated_terms.add(word)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    # 按照中文优先排序，英文按长度排序
    sorted_terms = sorted(
        associated_terms,
        key=lambda x: (0 if is_chinese(x) else 1, len(x) if not is_chinese(x) else 0),
    )

    # 返回前 5 个相关词语
    return sorted_terms[:10]


def is_chinese(word):
    """判断词语是否为中文"""
    return all("\u4e00" <= char <= "\u9fff" for char in word)
