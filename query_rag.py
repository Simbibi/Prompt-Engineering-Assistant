import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from settings import get_settings

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


async def generate_answer(question: str) -> str:
    try:
        logger.info(f"Начало генерации ответа для: {question}")
        # similarity_search — блокирующий
        retrieved_docs = await asyncio.to_thread(
            vector_store.similarity_search,
            question,
            3,
        )
        logger.info(f"Найдено {len(retrieved_docs)} документов")

        docs_content = "\n".join(doc.page_content for doc in retrieved_docs)

        message = prompt.invoke({
            "question": question,
            "context": docs_content,
        })

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
