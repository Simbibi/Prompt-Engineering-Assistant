import bs4
import sqlite3
import requests
from bs4 import BeautifulSoup
import asyncio

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from settings import get_settings
settings = get_settings()

def build_index():
    """Build and persist the vector index (synchronous).

    This is kept as a synchronous function because many used libraries are
    synchronous (requests, sqlite3, langchain sync clients). Importing this
    module will not run the blocking code; run it as a script or call
    `asyncio.to_thread(build_index)` from async code.
    """
    print(sqlite3.sqlite_version)
    #url = 'https://ru.wikipedia.org/wiki/Тяжёлые_металлы'
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
    print(len(docs))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)
    print("Number of chunks:", len(all_splits))
    for idx, chunk in enumerate(all_splits, 1):
        print(f"--- Chunk {idx} ---")
        print(chunk.page_content)
        print("\n")
    
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
    openai_api_base=settings.openai_api_url.get_secret_value() if settings.openai_api_url else None,
    )


    vector_store = Chroma(
        collection_name="prompt_engineering",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_db_dir),
    )

    ids = vector_store.add_documents(all_splits)
    print("Documents added:", len(ids))


if __name__ == "__main__":
    build_index()
