import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from query_rag import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='RAG', summary="A simple RAG application for answering questions.")

# Enable CORS so the frontend can call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the project root so index.html can be returned
app.mount("/static", StaticFiles(directory="."), name="static")


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


@app.get("/")
async def get_home():
    return FileResponse("index.html")