import logging
from fastapi import FastAPI
from query_rag import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title = 'RAG', summary="A simple RAG application for answering questions.")

@app.get("/ask")
async def ask(question: str):
    try:
        logger.info(f"Получен вопрос: {question}")
        answer = await generate_answer(question)
        logger.info("Ответ сгенерирован успешно")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500
