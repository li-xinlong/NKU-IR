import pandas as pd
from collections import defaultdict
import multiprocessing as mp


# 数据读取与预处理
def read_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    url_set = set(data["url"])  # 存在的 URL 集合
    graph = defaultdict(list)

    for _, row in data.iterrows():
        url = row["url"]
        links = row["links"]

        if pd.isna(links) or links.strip() == "":
            graph[url] = []  # 作为叶子节点
        else:
            valid_links = [link for link in links.split(";") if link in url_set]
            graph[url] = valid_links

    return graph, data


# 全局默认值初始化函数（避免 lambda）
def default_pr_value(num_nodes, damping_factor):
    return damping_factor / num_nodes


# 并行 PageRank 子任务
def pagerank_worker(task_data):
    graph, damping, global_pr, damping_factor, num_nodes, nodes = task_data
    local_pr = {node: global_pr[node] for node in nodes}

    # 初始化 PR 值的字典
    new_pr = {node: damping_factor / num_nodes for node in graph}

    for node in nodes:
        if not graph[node]:  # 处理叶子节点
            for other_node in graph:
                new_pr[other_node] += damping * (local_pr[node] / num_nodes)
        else:
            for out_node in graph[node]:
                new_pr[out_node] += damping * (local_pr[node] / len(graph[node]))

    return new_pr


# 并行计算 PageRank
def parallel_pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    num_nodes = len(graph)
    damping_factor = 1 - damping

    # 初始化 PR 值
    global_pr = {node: 1 / num_nodes for node in graph}

    nodes = list(graph.keys())
    num_workers = min(mp.cpu_count(), len(nodes))
    chunks = [nodes[i::num_workers] for i in range(num_workers)]

    for iteration in range(max_iter):
        with mp.Pool(num_workers) as pool:
            results = pool.map(
                pagerank_worker,
                [
                    (
                        graph,
                        damping,
                        global_pr.copy(),
                        damping_factor,
                        num_nodes,
                        chunk,
                    )
                    for chunk in chunks
                ],
            )

        # 合并子任务结果
        new_global_pr = {node: damping_factor / num_nodes for node in graph}
        for partial_pr in results:
            for node, value in partial_pr.items():
                new_global_pr[node] += value

        # 收敛判断
        diff = sum(
            abs(new_global_pr[node] - global_pr.get(node, 0)) for node in new_global_pr
        )
        print(f"Iteration {iteration + 1}/{max_iter}, Difference: {diff:.6f}")
        if diff < tol:
            print("Convergence achieved!")
            return dict(new_global_pr)

        # 更新全局 PR
        global_pr = dict(new_global_pr)

    print("Max iterations reached without convergence.")
    return dict(global_pr)


# 结果保存
def save_results(data, pr, output_file):
    data["pagerank"] = data["url"].map(pr).fillna(0)
    result = data[["url", "pagerank"]]
    result.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# 主程序
if __name__ == "__main__":
    # 输入文件路径
    file_path = "../crawler/title_url_anchor_body.csv"
    output_file = "pagerank_results.csv"

    try:
        # 数据读取与预处理
        graph, data = read_and_preprocess(file_path)
        print("Data preprocessing completed.")

        # 并行计算 PageRank
        final_pr = parallel_pagerank(graph)
        print("PageRank computation completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        final_pr = None

    finally:
        if final_pr:
            save_results(data, final_pr, output_file)
        else:
            print("PageRank computation failed; no results to save.")
