# -*- coding: utf-8 -*-
import argparse
import requests
import re
import os
import time

import arxiv


XIAOMI_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
XIAOMI_API_KEY = os.getenv('API_KEY')
XIAOMI_MODEL = "mimo-v2.5"


def translate_text_xiaomi(text):
    if not text or not text.strip():
        return ""
    if not XIAOMI_API_KEY:
        print("警告: 未设置API_KEY，跳过翻译")
        return text
    try:
        resp = requests.post(
            XIAOMI_API_BASE + "/chat/completions",
            headers={"Authorization": "Bearer " + XIAOMI_API_KEY, "Content-Type": "application/json"},
            json={
                "model": XIAOMI_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的学术论文翻译助手。请将用户提供的英文论文标题和摘要翻译成中文，保持学术专业性，不要添加任何解释或注释。"},
                    {"role": "user", "content": "请翻译以下内容：\n\n" + text}
                ],
                "temperature": 0.3
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("翻译失败: " + str(e))
        return text


def get_authors(authors, first_author=False):
    return str(authors[0]) if first_author else "，".join(str(author) for author in authors)


def escape_markdown(text):
    md_chars = ["`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", "!", "|"]
    text = re.sub(r'(https?://)', r'URLPROTOCOL\g<1>', text)
    for char in md_chars:
        text = text.replace(char, "\\" + char)
    return re.sub(r'URLPROTOCOL(https?://)', r'\g<1>', text)


def _fetch_results(client, search, max_retries=5, initial_delay=60.0):
    """带指数退避重试的结果获取器，处理 arXiv API 429 限流"""
    for attempt in range(max_retries):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as e:
            if e.status != 429 or attempt == max_retries - 1:
                raise

            delay = initial_delay * (2 ** attempt)
            print(
                f"arXiv API 限流 (429)，等待 {delay:.0f} 秒后重试 "
                f"({attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)


def save_papers_to_md_file(query="YOLO", max_results=5, filename="README.md"):
    client = arxiv.Client(delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    md_content = [
        "<div align=\"center\">",
        "",
        "# YOLO ArXiv Daily",
        "",
        "[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()",
        "[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)",
        "[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)",
        "",
        "*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*",
        "",
        "</div>",
        "",
        "---",
        "",
        "## 📑 论文列表",
        ""
    ]

    for i, result in enumerate(_fetch_results(client, search), 1):
        urls_in_abstract = re.findall(r'(https?://[^\s]+)', result.summary)
        code_links = "，".join(urls_in_abstract) if urls_in_abstract else None

        abstract_no_newline = result.summary.replace('\n', ' ').replace('\\', '')

        title_zh = translate_text_xiaomi(result.title)
        abstract_zh = translate_text_xiaomi(abstract_no_newline)

        title = escape_markdown(result.title)
        authors = escape_markdown(get_authors(result.authors, first_author=True))
        publish_time = result.published.strftime('%Y-%m-%d')
        abstract = escape_markdown(abstract_no_newline)

        md_content.append("> ### " + str(i) + ". " + title)
        if title_zh and title_zh != result.title:
            md_content.append("> **🔹 中文标题：** " + title_zh)
        md_content.append(">")
        md_content.append("> | 属性 | 内容 |")
        md_content.append("> |:---:|:---|")
        md_content.append("> | 📅 发布日期 | " + publish_time + " |")
        md_content.append("> | 👤 作者 | " + authors + " |")
        md_content.append(">")
        md_content.append("> **📄 英文摘要：**")
        md_content.append("> " + abstract)
        if abstract_zh and abstract_zh != abstract_no_newline:
            md_content.append(">")
            md_content.append("> **📝 中文摘要：**")
            md_content.append("> " + abstract_zh)
        if code_links:
            md_content.append(">")
            md_content.append("> **💻 代码链接：** " + code_links)
        md_content.append(">")
        md_content.append("> 🔗 [阅读论文](" + result.entry_id + ")")
        md_content.append("")
        md_content.append("---")
        md_content.append("")

    md_content.extend([
        "<div align=\"center\">",
        "",
        "*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*",
        "",
        "</div>"
    ])

    with open(filename, "w", encoding='utf-8') as md_file:
        md_file.write("\n".join(md_content))
    print("Markdown内容已保存到 " + filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从arXiv获取论文并翻译摘要")
    parser.add_argument("-q", "--query", default="YOLO", help="搜索关键词 (默认: YOLO)")
    parser.add_argument("-n", "--num", type=int, default=5, help="获取论文数量 (默认: 5)")
    parser.add_argument("-o", "--output", default="README.md", help="输出文件 (默认: README.md)")
    args = parser.parse_args()
    save_papers_to_md_file(query=args.query, max_results=args.num, filename=args.output)
