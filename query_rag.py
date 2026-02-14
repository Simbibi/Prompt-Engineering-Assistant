import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from api import OPENAI_API_KEY, OPENAI_API_BASE
from settings import settings

# инициализация (один раз)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.api_key,
    openai_api_base=settings.llm_vender_secret,
)

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

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
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)


async def generate_answer(question: str) -> str:
    # similarity_search — блокирующий
    retrieved_docs = await asyncio.to_thread(
        vector_store.similarity_search,
        question,
        3,
    )

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

    return response.content
