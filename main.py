import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from settings import get_settings
from collections import defaultdict
import math
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
settings = get_settings()

# инициализация (один раз)
try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
        openai_api_base=settings.openai_api_url.get_secret_value() if settings.openai_api_url else None,
    )
    logger.info("OpenAI Embeddings инициализирована успешно")
except Exception as e:
    logger.error(f"Ошибка при инициализации OpenAI Embeddings: {e}")
    raise

try:
    vector_store = Chroma(
        collection_name="prompt_engineering",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_db_dir),
    )
    logger.info("Chroma vector store инициализирована успешно")
except Exception as e:
    logger.error(f"Ошибка при инициализации Chroma: {e}")
    raise

try:
    data = vector_store.get(include=["documents", "metadatas"])
    docs = [
        Document(
            page_content=doc,
            metadata=meta,
        )
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 10
    logger.info("BM25 retriever инициализирована успешно")
except Exception as e:
    logger.error(f"Ошибка при инициализации BM25: {e}")
    raise

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions about the blog post on prompt engineering.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say "I don't know."
Question: {question}
Context: {context}
Answer:"""
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
    openai_api_base=settings.openai_api_url.get_secret_value() if settings.openai_api_url else None,
)

def rrf_fusion(dense_docs, sparse_docs, fusion_k=60):
    scores = defaultdict(float)
    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata["chunk_id"]
        scores[doc_id] += 1 / (fusion_k + rank + 1)
    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.metadata["chunk_id"]
        scores[doc_id] += 1 / (fusion_k + rank + 1)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_scores]

async def generate_answer(question: str) -> str:
    try:
        logger.info(f"Начало генерации ответа для: {question}")
        # similarity_search — блокирующий (dense retrieval)
        dense_docs = await asyncio.to_thread(
            vector_store.similarity_search,
            question,
            10,  # больше для лучшей fusion
        )
        # BM25 retrieval — блокирующий (sparse retrieval)
        sparse_docs = await asyncio.to_thread(
            bm25.invoke,
            question,
        )
        # Hybrid fusion с RRF
        fused_chunk_ids = rrf_fusion(dense_docs, sparse_docs)
        # Берём top-k (как в исходном коде k=3)
        top_k = 3
        fused_chunk_ids = fused_chunk_ids[:top_k]
        # Собираем уникальные документы по chunk_id
        all_docs_map = {doc.metadata["chunk_id"]: doc for doc in dense_docs + sparse_docs}
        retrieved_docs = [all_docs_map[chunk_id] for chunk_id in fused_chunk_ids if chunk_id in all_docs_map]
        logger.info(f"Найдено {len(retrieved_docs)} документов")
        docs_content = "\n".join(doc.page_content for doc in retrieved_docs)
        # prompt.invoke — блокирующий
        message = await asyncio.to_thread(
            prompt.invoke,
            {"question": question, "context": docs_content},
        )
        # llm.invoke — блокирующий
        response = await asyncio.to_thread(
            llm.invoke,
            message,
        )
        logger.info("Ответ успешно получен от LLM")
        return response.content
    except Exception as e:
        logger.error(f"Ошибка в generate_answer: {e}", exc_info=True)
        raise