import re
import arxiv
import os

# Read the value of PERSON_NAME environment variable
person_name = os.getenv('PERSON_NAME', 'World')

# Use person_name in your script
print(f"Hello, {person_name}!")

def get_authors(authors, first_author=False):
    """Return a formatted string of authors."""
    return str(authors[0]) if first_author else ", ".join(str(author) for author in authors)

def escape_markdown(text):
    """Escape markdown special characters."""
    md_chars = ["\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!"]
    for char in md_chars:
        text = text.replace(char, "\\" + char)
    return text

def save_papers_to_md_file(query="YOLO", max_results=5, filename="yolo_papers.md"):
    """Fetch daily papers from arXiv based on the given query and save to Markdown file."""
    md_content = ["# Daily YOLO Papers from arXiv\n\n"]
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    for result in search.results():
        title = escape_markdown(result.title)
        url = result.entry_id
        authors = escape_markdown(get_authors(result.authors, first_author=True))
        publish_time = result.published.date()
        abstract = escape_markdown(result.summary)

        urls_in_abstract = re.findall(r'(https?://[^\s]+)', abstract)
        code_links = ", ".join(urls_in_abstract) if urls_in_abstract else "No code link found in abstract."

        md_content.append(f"## {title}\n")
        md_content.append(f"**Published Date**: {publish_time}\n")
        md_content.append(f"**Authors**: {authors}\n")
        md_content.append(f"**Abstract**: {abstract}\n\n")
        md_content.append(f"**Code Links**: {code_links}\n")
        md_content.append(f"**Paper URL**: [Read More]({url})\n")
        md_content.append("---\n\n")

    with open(filename, "w") as md_file:
        md_file.write("\n".join(md_content))
    print(f"Markdown content saved to {filename}")

if __name__ == "__main__":
    save_papers_to_md_file(query="YOLO", max_results=10, filename="README.md")
