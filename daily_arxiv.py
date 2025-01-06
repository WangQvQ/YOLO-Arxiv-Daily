import time
import requests
import random
import hashlib
import re
import datetime

# 假设你已经安装并可以导入arxiv
import arxiv

# 百度翻译API配置，请替换为你自己的凭证
import os

appid = os.getenv('BAI_DU_APPID')
appkey = os.getenv('BAI_DU_APPKEY')
from_lang = 'en'
to_lang = 'zh'

def translate_text_baidu(query):
    """利用百度翻译API翻译文本"""
    time.sleep(1)  # 在发送请求前等待1秒，以避免过于频繁的请求
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    salt = random.randint(32768, 65536)
    sign = hashlib.md5((appid + query + str(salt) + appkey).encode('utf-8')).hexdigest()

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {
        'appid': appid,
        'q': query,
        'from': from_lang,
        'to': to_lang,
        'salt': salt,
        'sign': sign
    }

    response = requests.post(url, data=payload, headers=headers)
    result = response.json()

    if 'trans_result' in result:
        translations = [trans['dst'] for trans in result['trans_result']]
        return "\n".join(translations)  # 将翻译结果重新组合成一个长字符串，各段落以换行符分隔
    else:
        print("翻译失败，错误信息：", result.get("error_msg", "未知错误"))
        return ""

def get_authors(authors, first_author=False):
    """返回作者字符串。"""
    return str(authors[0]) if first_author else "，".join(str(author) for author in authors)

def escape_markdown(text):
    """转义Markdown特殊字符，不包括URLs。"""
    md_chars = ["`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", "!", "|"]
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

        # 消除摘要中的换行和转义反斜杠
        abstract_no_newline = result.summary.replace('\n', ' ').replace('\\', '')
        
        title_zh = translate_text_baidu(result.title)
        abstract_zh = translate_text_baidu(abstract_no_newline)  # 使用处理过的摘要进行翻译

        title = escape_markdown(result.title)
        authors = escape_markdown(get_authors(result.authors, first_author=True))
        publish_time = result.published.strftime('%Y-%m-%d')
        abstract = escape_markdown(abstract_no_newline)  # 使用处理过的摘要

        md_content.append(f"## {title} / {title_zh}\n")
        md_content.append(f"发布日期：{publish_time}\n")
        md_content.append(f"作者：{authors}\n")
        md_content.append(f"摘要：{abstract}\n")  # 显示处理过的摘要
        md_content.append(f"中文摘要：{abstract_zh}\n\n")
        md_content.append(f"代码链接：{code_links}\n")
        md_content.append(f"论文链接：[阅读更多]({result.entry_id})\n")
        md_content.append("---\n\n")

    with open(filename, "w", encoding='utf-8') as md_file:
        md_file.write("\n".join(md_content))
    print(f"Markdown内容已保存到 {filename}")




if __name__ == "__main__":
    save_papers_to_md_file(query="YOLO", max_results=5, filename="README.md")
