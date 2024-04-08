import re
import arxiv

def get_authors(authors, first_author=False):
    """返回作者字符串。"""
    return str(authors[0]) if first_author else "，".join(str(author) for author in authors)

def escape_markdown(text):
    """转义Markdown特殊字符，不包括URLs。"""
    md_chars = ["\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", "!", "|"]
    text = re.sub(r'(https?://)', r'URLPROTOCOL\g<1>', text)
    for char in md_chars:
        text = text.replace(char, "\\" + char)
    return re.sub(r'URLPROTOCOL(https?://)', r'\g<1>', text)

def save_papers_to_md_file(query="YOLO", max_results=5, filename="README.md"):
    """根据给定查询从arXiv获取论文，并将其保存到Markdown文件。"""
    md_content = ["# 每日从arXiv中获取最新YOLO相关论文\n\n"]
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    for result in search.results():
        urls_in_abstract = re.findall(r'(https?://[^\s]+)', result.summary)
        code_links = "，".join(urls_in_abstract) if urls_in_abstract else "摘要中未找到代码链接。"

        title = escape_markdown(result.title)
        authors = escape_markdown(get_authors(result.authors, first_author=True))
        publish_time = result.published.date()
        abstract = escape_markdown(result.summary)

        md_content.append(f"## {title}\n")
        md_content.append(f"**发布日期**：{publish_time}\n")
        md_content.append(f"**作者**：{authors}\n")
        md_content.append(f"**摘要**：{abstract}\n\n")
        md_content.append(f"**代码链接**：{code_links}\n")
        md_content.append(f"**论文链接**：[阅读更多]({result.entry_id})\n")
        md_content.append("---\n\n")

    with open(filename, "w", encoding='utf-8') as md_file:
        md_file.write("\n".join(md_content))
    print(f"Markdown内容已保存到 {filename}")

if __name__ == "__main__":
    save_papers_to_md_file(query="YOLO", max_results=10, filename="README.md")
