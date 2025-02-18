import os
import json
import pandas as pd
from collections import defaultdict
import csv
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from PyPDF2 import PdfReader
import docx
from urllib.request import urlretrieve
import wget
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from term_association_search import search_associated_terms

# 文件路径配置
TF_IDF_DIR = "../indexer/tf_idf_chunks"
Title_TF_IDF_DIR = "../indexer/title_tf_idf_chunks"
File_TF_IDF_DIR = "../indexer/file_tf_idf_chunks"

PAGERANK_FILE = "../pagerank/pagerank_results.csv"
DOC_MAPPING_FILE = "../crawler/linenumber_title_url_anchor_body.csv"
QUERY_LOG_FILE = "query_log.txt"
RESULT_FILE = "result.txt"
PAGE_PHOTOS_DIR = "page_photos"  # 网页快照保存的文件夹
# 支持的文件类型
SUPPORTED_FILE_FORMATS = [".pdf", ".doc", ".docx", ".xls", ".xlsx"]

# 创建页面快照保存的文件夹
if not os.path.exists(PAGE_PHOTOS_DIR):
    os.makedirs(PAGE_PHOTOS_DIR)


def load_file_preview(file_url):
    """
    加载文件内容并返回前 30 行的文字。

    参数:
        file_url (str): 文件的 URL。

    返回:
        str: 文件的前 30 行文字，若读取失败返回空字符串。
    """
    try:
        # 下载文件
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # 确定文件类型
        file_extension = file_url.split(".")[-1].lower()

        if file_extension == "pdf":
            return load_pdf_preview(response.content)
        elif file_extension in ["doc", "docx"]:
            return load_doc_preview(response.content)
        elif file_extension in ["xls", "xlsx"]:
            return load_excel_preview(response.content)
        elif file_extension == "txt":
            return load_text_preview(response.text)
        else:
            # print(f"不支持的文件格式：{file_extension}")
            return "[ERROR: 不支持的文件格式]"
    except Exception as e:
        # print(f"无法加载文件预览: {e}")
        return "[ERROE: 无法加载文件预览]"


def load_pdf_preview(content):
    """解析 PDF 文件并返回前 30 行文字。"""
    try:
        from io import BytesIO

        pdf = PdfReader(BytesIO(content))
        lines = []
        for page in pdf.pages:
            lines.extend(page.extract_text().splitlines())
            if len(lines) >= 30:
                break
        return "\n".join(lines[:30])
    except Exception as e:
        # print(f"PDF 解析失败: {e}")
        return ""


def load_doc_preview(content):
    """解析 Word 文档并返回前 30 行文字。"""
    try:
        from io import BytesIO

        doc = docx.Document(BytesIO(content))
        lines = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(lines[:30])
    except Exception as e:
        # print(f"Word 文档解析失败: {e}")
        return ""


def load_excel_preview(content):
    """解析 Excel 文件并返回前 30 行文字。"""
    try:
        from io import BytesIO

        excel_data = pd.read_excel(BytesIO(content))
        lines = []
        for index, row in excel_data.iterrows():
            lines.append(", ".join(map(str, row.values)))
            if len(lines) >= 30:
                break
        return "\n".join(lines)
    except Exception as e:
        # print(f"Excel 文件解析失败: {e}")
        return ""


def load_text_preview(content):
    """解析纯文本文件并返回前 30 行文字。"""
    return "\n".join(content.splitlines()[:30])


# 加载 PageRank 数据
def load_pagerank_data(pagerank_file):
    """加载 PageRank 数据"""
    pagerank_data = {}
    with open(pagerank_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            pagerank_data[row[0]] = float(row[1])
    return pagerank_data


# 获取查询词对应的TF-IDF索引文件
def get_tf_idf_file_for_term(term):
    """根据查询词确定对应的TF-IDF文件"""
    if "\u4e00" <= term[0] <= "\u9fff":  # 中文字符
        return f"chinese_{term[0]}.json"
    elif term[0].isalpha():  # 英文字母
        return f"alpha_{term[0].lower()}.json"
    elif term[0].isdigit():  # 数字
        return "numeric.json"
    else:  # 其他符号
        return "others.json"


# 通配符转换为正则表达式
def wildcard_to_regex(term):
    """将通配符查询转换为正则表达式"""
    regex = re.escape(term).replace(r"\*", ".*").replace(r"\?", ".")
    return f"^{regex}$"  # 完整匹配


# 动态加载查询词对应的TF-IDF文件并支持通配符查询
def load_tf_idf_for_terms(terms, tf_idf_dir):
    """加载查询词对应的TF-IDF文件，并支持通配符查询"""
    loaded_files = {}
    term_to_doc_tf_idf = defaultdict(dict)
    all_doc_ids = None  # 用于存储所有查询词的交集文档ID

    for term in terms:
        regex = re.compile(wildcard_to_regex(term))  # 转换为正则表达式
        file_name = get_tf_idf_file_for_term(term)
        file_path = os.path.join(tf_idf_dir, file_name)

        if file_path not in loaded_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_files[file_path] = json.load(f)
            except FileNotFoundError:
                loaded_files[file_path] = {}

        current_doc_ids = set()
        for candidate_term, tf_idf_data in loaded_files[file_path].items():
            if regex.match(candidate_term):  # 匹配符合正则的词
                for doc_id, tf_idf in tf_idf_data:
                    term_to_doc_tf_idf[term][doc_id] = tf_idf
                    current_doc_ids.add(doc_id)

        # 求交集
        if all_doc_ids is None:
            all_doc_ids = current_doc_ids
        else:
            all_doc_ids &= current_doc_ids

    # 过滤掉不在交集中的文档
    if all_doc_ids is not None:
        for term in term_to_doc_tf_idf:
            term_to_doc_tf_idf[term] = {
                doc_id: tf_idf
                for doc_id, tf_idf in term_to_doc_tf_idf[term].items()
                if doc_id in all_doc_ids
            }

    return term_to_doc_tf_idf


# 计算文档得分
def compute_document_scores(term_to_doc_tf_idf, pagerank_scores):
    """根据TF-IDF值和PageRank分数计算文档总得分"""
    doc_scores = defaultdict(float)

    for term, doc_tf_idf in term_to_doc_tf_idf.items():
        for doc_id, tf_idf in doc_tf_idf.items():
            pagerank = pagerank_scores.get(doc_id, 0)
            doc_scores[doc_id] += tf_idf * pagerank

    return doc_scores


def compute_document_scores_history(
    term_to_doc_tf_idf, pagerank_scores, history_doc_tf_idf
):
    """根据TF-IDF值、PageRank分数以及历史记录计算文档总得分"""
    doc_scores = defaultdict(float)

    # 获取全局词汇表
    vocab = set()
    for tf_idf in term_to_doc_tf_idf.values():
        vocab.update(tf_idf.keys())
    for tf_idf in history_doc_tf_idf.values():
        vocab.update(tf_idf.keys())
    vocab = sorted(vocab)

    # 向量化方法
    def vectorize(tf_idf, vocab):
        return np.array([tf_idf.get(term, 0) for term in vocab])

    # 构建历史记录向量
    if history_doc_tf_idf:
        history_vector = np.sum(
            [vectorize(tf_idf, vocab) for tf_idf in history_doc_tf_idf.values()], axis=0
        )
        history_norm = np.linalg.norm(history_vector)
        if history_norm > 0:
            history_vector /= history_norm
        else:
            history_vector = None
    else:
        history_vector = None

    # 遍历文档TF-IDF数据
    for term, doc_tf_idf in term_to_doc_tf_idf.items():
        for doc_id, tf_idf in doc_tf_idf.items():
            # 获取文档PageRank分数
            pagerank = pagerank_scores.get(doc_id, 0)

            # 构建文档向量
            doc_vector = vectorize(doc_tf_idf, vocab)
            doc_norm = np.linalg.norm(doc_vector)
            if doc_norm > 0:
                doc_vector /= doc_norm

            # 计算文档与历史记录的余弦相似度
            if history_vector is not None:
                cosine_similarity = np.dot(history_vector, doc_vector)
            else:
                cosine_similarity = 0  # 无历史记录则相似度为0

            # 综合得分计算
            doc_scores[doc_id] += tf_idf * pagerank * (1 + cosine_similarity)

    return doc_scores


# 提取文档内容
def extract_document_content(doc_id, doc_mapping):
    """提取文档内容的前三行"""
    row = doc_mapping[doc_mapping["line_number"] == int(doc_id)]
    if row.empty:
        return None, None

    url = row.iloc[0]["url"]
    body = row.iloc[0]["body"]

    # 如果是文件路径，尝试读取文件内容
    if os.path.exists(body) and any(
        body.endswith(ext) for ext in SUPPORTED_FILE_FORMATS
    ):
        content = extract_file_content(body)
    else:
        content = body

    return url, "\n".join(content.splitlines()[:3]) if content else "[No Content]"


def extract_file_content(file_path):
    """提取文件内容（简化示例，仅支持TXT文件）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"


# 保存网页快照
def save_webpage_snapshot(url, folder=PAGE_PHOTOS_DIR):
    """保存网页快照，并从请求的网页中提取正文保存为文本文件"""
    try:
        # 获取网页内容
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败会抛出异常

        # 使用BeautifulSoup解析网页
        soup = BeautifulSoup(response.text, "html.parser")

        # 清理HTML中的脚本、样式等
        for script in soup(["script", "style"]):
            script.decompose()

        # 获取网页标题并生成文件名
        title = soup.title.string if soup.title else "snapshot"
        # 使用 URL 作为文件夹名
        url_parsed = urlparse(url)
        folder_name = os.path.join(folder, url_parsed.netloc)
        os.makedirs(folder_name, exist_ok=True)  # 创建以网址为名字的文件夹

        # 保存网页HTML快照
        html_filename = f"{title}.html"
        html_filepath = os.path.join(folder_name, html_filename)
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"网页快照已保存：{html_filepath}")

        # 提取网页正文内容
        # 假设正文内容在 <body> 标签内，或其他具体标签内（如 <article> 或 <div class="content">）
        body_content = soup.find(
            "body"
        )  # 获取 <body> 标签的内容，您可以根据页面结构调整

        if body_content:
            # 获取正文文本（去除标签）
            body_text = body_content.get_text(separator="\n", strip=True)

            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 将正文保存为txt文件，格式如下：
            # 第一行保存时的时间
            # 第二行 title:
            # 第三行 body:
            txt_filename = f"{url_parsed.netloc}.txt"
            txt_filepath = os.path.join(folder_name, txt_filename)

            with open(txt_filepath, "w", encoding="utf-8-sig") as f:
                f.write(f"时间: {current_time}\n")
                f.write(f"title: {title}\n")
                f.write(f"body:\n{body_text}\n")

            print(f"正文内容已保存：{txt_filepath}")
        else:
            print(f"未找到网页正文内容：{url}")

    except Exception as e:
        print(f"保存网页快照失败: {e}")


# 保存查询结果到 result.txt
def save_query_results(results, result_file=RESULT_FILE):
    """将查询结果保存到 result.txt（覆盖写入）"""
    with open(result_file, "w", encoding="utf-8") as f:
        for idx, result in enumerate(results, start=1):
            f.write(f"Result {idx}:\n")
            f.write(f"URL: {result['url']}\n")
            if result["preview"] is not None:
                f.write(f"Preview: {result['preview']}\n")
            f.write("-" * 50 + "\n")


# 保存查询日志
def save_query_log(query_terms, urls, log_file=QUERY_LOG_FILE):
    """将查询词和返回的 URL 保存到查询日志文件"""
    with open(log_file, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(f"[{' '.join(query_terms)}]: {url}\n")


# 读取最近的查询词
def get_recent_queries(log_file=QUERY_LOG_FILE, num_queries=5):
    """从查询日志中读取最近不相同的查询词"""
    if not os.path.exists(log_file):
        return []  # 如果日志文件不存在，返回空列表

    queries = []
    seen_queries = set()  # 用于跟踪已经出现过的查询词
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[") and "]: " in line:  # 日志格式 [查询词]: URL
                query_terms = line.split("]:")[0][1:]  # 提取 [查询词] 的内容

                # 如果该查询词未出现过，则添加到查询词列表
                if query_terms not in seen_queries:
                    seen_queries.add(query_terms)
                    queries.append(query_terms)

    return queries[-num_queries:]  # 返回最近不相同的查询词


# 主查询函数
def query_documents(
    query_terms,
    tf_dif_dir,
    recent_queries,
):
    """执行查询"""
    pagerank_data = load_pagerank_data(PAGERANK_FILE)
    doc_mapping = pd.read_csv(DOC_MAPPING_FILE)

    # 加载 TF-IDF 数据
    term_to_doc_tf_idf = load_tf_idf_for_terms(query_terms, tf_dif_dir)
    history_doc_tf_idf = load_tf_idf_for_terms(recent_queries, tf_dif_dir)
    # print(term_to_doc_tf_idf)
    # print(history_doc_tf_idf)
    # 计算文档得分
    if history_doc_tf_idf:
        doc_scores = compute_document_scores_history(
            term_to_doc_tf_idf, pagerank_data, history_doc_tf_idf
        )
    else:
        doc_scores = compute_document_scores(term_to_doc_tf_idf, pagerank_data)

    # 排序文档得分
    sorted_doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # 提取查询结果
    results = []
    for rank, (doc_id, score) in enumerate(sorted_doc_scores):
        if rank < 5:  # 前5个结果计算 preview
            url, content_preview = extract_document_content(doc_id, doc_mapping)
            if is_file_type(url):
                content_preview = load_file_preview(url)
        else:  # 第5个之后 preview 设置为 None
            row = doc_mapping[doc_mapping["line_number"] == int(doc_id)]
            if row.empty:
                url = None
            url = row.iloc[0]["url"]
            content_preview = None
        if url:
            results.append({"url": url, "preview": content_preview})

    # 提取 URL 列表并保存到查询日志
    urls = [result["url"] for result in results]
    save_query_log(query_terms, urls[:5])
    # print(len(results))
    # 保存到 result.txt
    save_query_results(results)

    return results


SUPPORTED_FILE_FORMATS = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"]


def is_file_type(url, supported_formats=SUPPORTED_FILE_FORMATS):
    """
    判断 URL 是否为文件类型。

    参数:
        url (str): 要判断的 URL。
        supported_formats (list): 支持的文件格式扩展名列表。

    返回:
        bool: 如果是文件类型则返回 True，否则返回 False。
    """
    try:
        file_extension = "." + url.rsplit(".", 1)[-1] if "." in url else ""
        return file_extension in supported_formats
    except Exception as e:
        print(f"判断文件类型时发生错误: {e}")
        return False


def download_file_and_generate_txt(url, folder):
    """
    下载文件并生成对应的 .txt 文件，记录文件保存时间和文件名。

    参数:
        url (str): 文件的 URL。
        folder (str): 保存文件的目标文件夹。

    返回:
        None
    """
    try:
        # 解析 URL
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)  # 提取文件名
        folder_name = os.path.join(folder, parsed_url.netloc)  # 以域名为文件夹
        os.makedirs(folder_name, exist_ok=True)  # 创建文件夹

        # 下载文件
        urlretrieve(url, file_path)  # 下载文件并保存
        file_path = os.path.join(folder_name, file_name)
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # 检查请求是否成功
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):  # 以 8KB 为块大小读取
                    f.write(chunk)
        print(f"文件已下载：{file_path}")
        # 生成对应的 .txt 文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txt_file_name = f"{os.path.splitext(file_name)[0]}.txt"
        txt_file_path = os.path.join(folder_name, txt_file_name)

        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(f"时间: {current_time}\n")
            f.write(f"链接: {url}\n")
            f.write(f"文件名: {file_name}\n")
            f.write("内容：文件已成功下载。")
        print(f"文件下载信息已保存：{txt_file_path}")

    except Exception as e:
        print(f"下载文件或生成信息文件失败: {e}")


def prompt_and_save_snapshots(results, max_attempts=3):
    """
    询问用户是否保存网页快照，若是，询问要保存快照的编号并进行验证。
    如果用户输入无效编号超过最大尝试次数，退出程序。

    参数:
        results (list): 查询结果，每个结果应包含 'url' 键。
        max_attempts (int): 最大重试次数，默认为 3。
    """
    save_snapshot = input("是否需要保存网页快照？(y/n): ").strip().lower()

    if save_snapshot == "y":
        # 显示结果并让用户选择要保存快照的编号
        print("\n查询结果：")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['url']}")

        attempt_count = 0  # 当前尝试次数
        while attempt_count < max_attempts:
            snapshot_number = input(
                "请输入要保存快照的编号（多个编号用空格分隔）: "
            ).strip()

            # 分割输入并确保编号合法
            try:
                snapshot_numbers = [int(num) for num in snapshot_number.split()]
                valid = True  # 标记是否所有编号都有效
                for num in snapshot_numbers:
                    if 1 <= num <= len(results):
                        print(f"保存快照：{results[num - 1]['url']}")
                        if is_file_type(results[num - 1]["url"]):
                            download_file_and_generate_txt(
                                results[num - 1]["url"], PAGE_PHOTOS_DIR
                            )
                            print("文件已下载并生成对应的 .txt 文件。")
                        else:
                            save_webpage_snapshot(results[num - 1]["url"])
                    else:
                        print(f"无效编号：{num}")
                        valid = False  # 发现无效编号，标记为无效
                if valid:
                    break  # 所有编号有效，跳出循环
            except ValueError:
                print("请输入有效的编号！")

            attempt_count += 1
            if attempt_count >= max_attempts:
                print("无效输入超过最大尝试次数，程序将退出。")
                break


def select_query_type():
    """选择查询类型"""
    print("\n请选择查询类型:")
    print("1. 普通查询")
    print("2. 标题查询")
    print("3. 文件查询")
    query_type = input("请输入对应数字 (1/2/3): ").strip()

    if query_type == "1":
        return TF_IDF_DIR
    elif query_type == "2":
        return Title_TF_IDF_DIR
    elif query_type == "3":
        return File_TF_IDF_DIR
    elif query_type.upper() == "<EXIT>":
        print("程序已退出。")
        exit()
    else:
        print("无效输入，请重新选择。")
        return select_query_type()


# 循环查询
if __name__ == "__main__":
    print("请输入查询词（用空格分隔）。输入 <EXIT> 退出程序。")
    while True:
        recent_queries = get_recent_queries()
        if recent_queries:
            print("最近的查询词：")
            for idx, query in enumerate(recent_queries[::-1], start=1):
                print(f"{idx}. {query}")
        if_dif_dir = select_query_type()
        # print(f"当前查询类型: {if_dif_dir}")
        bo = False

        think_input = input("是否需要进行联想关联？(y/n):").strip().lower()
        if think_input.upper() == "<EXIT>":
            print("程序已退出。")
            break
        if think_input == "y":
            think_input = input("请输入联想词: ").strip()
            associated_terms = search_associated_terms(think_input, if_dif_dir)
            if associated_terms:
                print(f"与'{think_input}'相关的联想词：")
                for i, associated_term in enumerate(associated_terms, 1):
                    end_char = "\n" if i % 5 == 0 else ", "
                    print(associated_term, end=end_char)
                    # 确保最后一行不会以逗号结尾
                    if len(associated_terms) % 5 != 0:
                        print()  # 补充换行
        while True:
            user_input = input("查询词: ").strip()
            if user_input.upper() == "<EXIT>":
                print("程序已退出。")
                bo = True
                break
            query_terms = user_input.split()
            if not query_terms:
                print("请输入有效的查询词。")
                continue
            else:
                break
        if bo:
            break
        results = query_documents(query_terms, if_dif_dir, recent_queries)

        if not results:
            print("未找到匹配的结果。")
        else:
            for idx, result in enumerate(results[:5], start=1):  # 仅迭代前5个结果
                print(f"Result {idx}:")
                print(f"URL: {result['url']}")
                print("-" * 50)
            prompt_and_save_snapshots(results[:5])  # 仅传递前5个结果给后续处理
        print("-" * 50)
