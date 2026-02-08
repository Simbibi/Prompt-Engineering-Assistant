import bs4
import sqlite3
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from api import OPENAI_API_KEY, OPENAI_API_BASE

print(sqlite3.sqlite_version)

url = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
html_doc = requests.get(url).text
soup = BeautifulSoup(html_doc, "html.parser")
print(soup.prettify()[:1000])

bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)

loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

ids = vector_store.add_documents(all_splits)
print("Documents added:", len(ids))
