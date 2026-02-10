from fastapi import FastAPI
from query_rag import generate_answer

app = FastAPI()

@app.get("/ask")
async def ask(question: str):
    answer = generate_answer(question)
    return {"answer": answer}
