import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


from src.generation.answerer import answer
from src.retrieval.retriever import retrieve_chunks


app = FastAPI(
     title="Company Policy RAG API",
    description="Ask questions about company policy documents",
    version="1.0.0"
)

class AskRequest(BaseModel):
    question: str


def verify_api_key(x_api_key: str = Header(default=None)):
    expected_key = os.getenv("RAG_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured on the server."
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(
    request: AskRequest,
    x_api_key: str = Header(default=None)
):
    verify_api_key(x_api_key)

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    chunks = retrieve_chunks(question, k=4)

    if not chunks or max(chunk["score"] for chunk in chunks) < 0.40:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found."
      )
    generated_answer = answer(question, k=4)

    sources = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        sources.append({
            "filename": metadata.get("filename"),
            "page": metadata.get("page"),
            "score": round(chunk.get("score", 0), 4)
        })

    return {
        "question": question,
        "answer": generated_answer,
        "sources": sources
    }