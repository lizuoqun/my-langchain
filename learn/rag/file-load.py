from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from rich import print as rprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

text_loader = TextLoader(file_path="../asset/load/01-langchain-utf-8.txt", encoding="utf-8")
text = text_loader.load()
print(text)
print(f"\n\n\n")

loader = CSVLoader(file_path="../asset/load/02-load.csv")
csv = loader.load()
print(csv)
print(type(csv[0]))
print(csv[0].page_content)
print(f"\n\n\n")

json_loader = JSONLoader(
    file_path="../asset/load/03-load.json",
    # jq_schema=".",  ## 提取所有字段
    jq_schema=".messages[].content",
    text_content=False  # 保持原始 JSON 结构，将提取的数据转换为JSON字符串存入page_content字段中
)
json = json_loader.load()
rprint(json)
print(f"\n\n\n")

loader = PyPDFLoader(
    # 文件路径，支持本地文件和在线文件链接
    # file_path="../asset/load/04-sample.pdf",
    file_path="https://arxiv.org/pdf/alg-geom/9202012",
    # 提取模式:控制如何从 PDF 文件中解析和提取文本结构。
    #   plain 提取文本，默认值
    #   layout 布局感知提取模式，通常会通过插入大量的空格、换行符，来模拟原文档中的多栏、    缩进和间距（适用场景：学术论文（如arXiv论文）、多栏报刊杂志、带有左右分栏的合同）
    extraction_mode="plain",
)

pdf = loader.load()
print(pdf)
print(f"\n\n\n")

loader = UnstructuredWordDocumentLoader(
    # 文件路径
    file_path="../asset/load/05-sgg_chat.docx",
    # 加载模式:
    #   single 返回单个Document对象
    #   elements 按标题等元素切分文档
    mode="single",
)
word = loader.load()
print(word)
print(f"\n\n\n")

loader = UnstructuredMarkdownLoader(
    file_path="../asset/load/06-load.md",
    # 加载模式:
    #   single 返回单个Document对象
    #   elements 按标题等元素切分文档
    mode="single",
    # 解析策略：
    #   "fast"（快速模式），它会以最快的速度提取文本，不进行复杂的版面分析
    #   "hi_res" 高分辨率模式
    strategy="fast"
)
md = loader.load()
print(md)
print(f"\n\n\n")

loader = UnstructuredHTMLLoader(
    file_path="../asset/load/07-load.html",
    mode="elements",
    strategy="fast"
)
htmls = loader.load()
for html in htmls:
    print(html)
print(f"\n\n\n")
