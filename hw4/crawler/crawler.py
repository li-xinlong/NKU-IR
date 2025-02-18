import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv
import os

USER_AGENT = "NKU Crawler"
BASE_URL = "https://www.nankai.edu.cn"
FILE_DOWNLOAD_DIR = "downloads"
TIMEOUT = aiohttp.ClientTimeout(total=60, connect=60, sock_connect=60, sock_read=60)
BATCH_SIZE = 2000  # 每批并发的请求数
SUPPORTED_FILE_TYPES = [
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]


def ensure_download_dir():
    """
    确保文件下载目录存在
    """
    if not os.path.exists(FILE_DOWNLOAD_DIR):
        os.makedirs(FILE_DOWNLOAD_DIR)


async def fetch_and_save_file(session, url, content_type):
    """
    下载文件并保存到本地，返回相对路径
    """
    try:
        file_extension = {
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }.get(content_type, ".bin")
        filename = os.path.join(
            FILE_DOWNLOAD_DIR,
            os.path.basename(urlparse(url).path) or f"file{file_extension}",
        )
        async with session.get(url, timeout=TIMEOUT) as response:
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(await response.read())
        return filename
    except Exception as e:
        # print(f"文件下载失败: {e}")
        return None


async def fetch_page(session, url):
    """
    异步获取网页内容，同时处理编码问题和非 HTML 内容
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        async with session.get(url, headers=headers, timeout=TIMEOUT) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                html = await response.read()
                encoding = response.get_encoding() or "utf-8-sig"
                return {
                    "type": "html",
                    "content": html.decode(encoding, errors="ignore"),
                }
            elif any(ft in content_type for ft in SUPPORTED_FILE_TYPES):
                file_path = await fetch_and_save_file(session, url, content_type)
                return {"type": "file", "content": file_path}
            else:
                return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None


def parse_page(html_or_file, base_url):
    """
    根据内容类型解析 HTML 或记录文件路径
    """
    if html_or_file["type"] == "html":
        html = html_or_file["content"]
        soup = BeautifulSoup(html, "html.parser")
        title = (
            soup.title.string.strip() if soup.title and soup.title.string else "无标题"
        )
        body = soup.get_text(separator="\n").strip() if soup.body else "无正文"
        body = " ".join(body.split()) if body else "无正文"
        links = []
        anchor_texts = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href and is_valid_url(href):
                full_url = urljoin(base_url, href)
                links.append(full_url)
                anchor_text = a.get_text(strip=True)
                anchor_texts.append(" ".join(anchor_text.split()))
        return {
            "type": "html",
            "title": title,
            "body": body,
            "links": links,
            "anchor_texts": anchor_texts,
        }
    elif html_or_file["type"] == "file":
        return {
            "type": "file",
            "title": "无标题",
            "body": html_or_file["content"],
            "links": [],
            "anchor_texts": [],
        }
    return None


def is_valid_url(url):
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)
    except Exception:
        return False


def save_to_csv_with_links(data, filename, write_header=False):
    """
    将爬取的数据（包括链接）保存到 CSV 文件
    """
    try:
        mode = "a" if not write_header else "w"
        with open(filename, mode, newline="", encoding="utf-8") as csvfile:
            fieldnames = ["title", "url", "anchor_texts", "body", "links"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(data)
        print(f"已追加 {len(data)} 条记录到 {filename}")
    except IOError as e:
        print(f"保存到 CSV 文件失败: {e}")


async def crawl(start_url, max_depth=50, max_records=150000, report_interval=1000):
    visited = set()
    to_visit = [(start_url, 0)]
    crawled_data = []
    output_file = "title_url_anchor_body.csv"
    written_records = 0

    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        ensure_download_dir()
        save_to_csv_with_links([], output_file, write_header=True)

        while to_visit:
            if written_records >= max_records:
                print(f"已达到最大记录数限制：{max_records}条，停止爬取")
                break

            batch = []
            while to_visit and len(batch) < BATCH_SIZE:
                url, depth = to_visit.pop(0)
                if url not in visited and depth <= max_depth:
                    visited.add(url)
                    batch.append((url, depth))

            tasks = [fetch_page(session, url) for url, _ in batch]
            results = await asyncio.gather(*tasks)

            for i, html_or_file in enumerate(results):
                if html_or_file:
                    url, depth = batch[i]
                    data = parse_page(html_or_file, BASE_URL)
                    title = data["title"]
                    body = (
                        data["body"]
                        if data["type"] == "html"
                        else f"文件路径: {data['body']}"
                    )
                    anchor_texts = "; ".join(data["anchor_texts"])
                    links = "; ".join(data["links"])

                    crawled_data.append(
                        {
                            "title": title,
                            "url": url,
                            "anchor_texts": anchor_texts,
                            "body": body,
                            "links": links,
                        }
                    )

                    if len(crawled_data) % report_interval == 0:
                        print(f"已爬取记录数：{written_records + len(crawled_data)}")

                    if len(crawled_data) >= 6000:
                        save_to_csv_with_links(crawled_data, output_file)
                        written_records += len(crawled_data)
                        crawled_data.clear()

                    if written_records >= max_records:
                        print(f"已达到最大记录数限制：{max_records}条，停止爬取")
                        break

                    for link in data["links"]:
                        if link not in visited:
                            to_visit.append((link, depth + 1))

        if crawled_data and written_records < max_records:
            remaining_to_write = max_records - written_records
            save_to_csv_with_links(crawled_data[:remaining_to_write], output_file)
            written_records += len(crawled_data[:remaining_to_write])
            crawled_data.clear()

    print(f"爬取完成，总记录数：{written_records}")


if __name__ == "__main__":
    asyncio.run(crawl(BASE_URL))
