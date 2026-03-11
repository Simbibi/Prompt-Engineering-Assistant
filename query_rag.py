import asyncio
import logging
import re  

from rank_bm25 import BM25Okapi  

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# --- Embeddings ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
    openai_api_base=settings.openai_api_url.get_secret_value() if settings.openai_api_url else None,
)

# --- Vector store ---
vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_db_dir),
)

# =========================================================
# 🔥 NEW: ЗАГРУЖАЕМ ВСЕ ДОКУМЕНТЫ И СТРОИМ BM25
# =========================================================

all_data = vector_store.get()  # 🔥 NEW
all_texts = all_data["documents"]  # 🔥 NEW
all_metadatas = all_data["metadatas"]  # 🔥 NEW

all_docs = [  # 🔥 NEW
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_texts, all_metadatas)
]

logger.info(f"Загружено {len(all_docs)} документов из Chroma")  # 🔥 NEW


def tokenize(text):  # 🔥 NEW
    return re.findall(r"\w+", text.lower())


tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]  # 🔥 NEW
bm25 = BM25Okapi(tokenized_corpus)  # 🔥 NEW

logger.info("BM25 инициализирован успешно")  # 🔥 NEW

# =========================================================

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


# =========================================================
# RRF
# =========================================================

def rrf_fusion(rank_lists, k: int = 60):
    scores = {}

    for rank_list in rank_lists:
        for rank, chunk_id in enumerate(rank_list):
            score = 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================================================
# 🔥 NEW: НОРМАЛЬНЫЙ BM25 SEARCH
# =========================================================

async def bm25_search(question: str, top_k: int = 10):  # 🔥 NEW
    tokenized_query = tokenize(question)

    scores = await asyncio.to_thread(
        bm25.get_scores,
        tokenized_query
    )

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [all_docs[i] for i in top_indices]


# =========================================================
# GENERATE ANSWER
# =========================================================

async def generate_answer(question: str) -> str:
    try:
        logger.info(f"Начало генерации ответа для: {question}")

        # --- Dense search ---
        dense_docs = await asyncio.to_thread(
            vector_store.similarity_search,
            question,
            10,
        )

        # --- 🔄 UPDATED: BM25 search ---
        bm25_docs = await bm25_search(question, top_k=10)

        logger.info(
            f"Dense: {len(dense_docs)}, BM25: {len(bm25_docs)}"
        )

        # --- Получаем списки chunk_id ---
        dense_rank = [doc.metadata["chunk_id"] for doc in dense_docs]
        bm25_rank = [doc.metadata["chunk_id"] for doc in bm25_docs]

        # --- RRF ---
        fused = rrf_fusion([dense_rank, bm25_rank], k=60)

        # --- Берём top-3 после fusion ---
        top_chunk_ids = [chunk_id for chunk_id, _ in fused[:3]]

        # 🔄 UPDATED: безопасный mapping через all_docs
        id_to_doc = {
            doc.metadata["chunk_id"]: doc
            for doc in all_docs
        }

        retrieved_docs = [
            id_to_doc[cid]
            for cid in top_chunk_ids
            if cid in id_to_doc
        ]

        logger.info(f"После RRF выбрано {len(retrieved_docs)} документов")

        docs_content = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        message = prompt.invoke({
            "question": question,
            "context": docs_content
        })

        answer = await asyncio.to_thread(llm.invoke, message)

        return answer.content

    except Exception:
        logger.exception("Ошибка генерации ответа")
        raise