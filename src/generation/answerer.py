
import os
from pathlib import Path

import requests
import tiktoken
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval.retriever import retrieve_chunks
from src.retrieval.bm25_retriever import BM25Retriever


API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

SIMILARITY_THRESHOLD = 0.40
TOKEN_BUDGET = 3000

RETRIEVAL_CANDIDATES = 50

FALLBACK_ANSWER = "I don't have information on that"


# Load BM25 once
bm25_retriever = BM25Retriever()


def hybrid_retrieve(question, k=4):
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    Search the original question and its two parts when joined by 'and'.
    Remove duplicate text while preserving source metadata.
    """

    # Start with the original question
    queries = [question]

    # Split a compound question into two parts
    lower_question = question.lower()

    if " and " in lower_question:
        split_position = lower_question.find(" and ")

        first_part = question[:split_position].strip(" ?.,")
        second_part = question[split_position + 5:].strip(" ?.,?")

        if len(first_part) >= 8 and len(second_part) >= 8:
            queries.extend([first_part, second_part])

    scores = {}
    documents = {}

    # Search each query using vector and BM25 retrieval
    for search_query in queries:

        vector_results = retrieve_chunks(
            search_query,
            k=RETRIEVAL_CANDIDATES
        )

        bm25_results = bm25_retriever.retrieve(
            search_query,
            k=RETRIEVAL_CANDIDATES
        )

        # Add vector rankings to RRF
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (60 + rank)
            )

            if doc_id not in documents:
                documents[doc_id] = result.copy()
            elif result["score"] > documents[doc_id]["score"]:
                documents[doc_id]["score"] = result["score"]

        # Add BM25 rankings to RRF
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (60 + rank)
            )

            # Preserve vector result metadata if already available
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "text": result["text"],
                    "score": 0.0,
                    "metadata": {
                        "filename": result["filename"],
                        "department": result["department"],
                        "page": result["page"]
                    }
                }

    # Sort documents by combined RRF score
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    final_results = []
    seen_text = set()

    for doc_id in ranked_ids:
        result = documents[doc_id]

        text = " ".join(result["text"].lower().split())

        # Avoid duplicate content
        if text in seen_text:
            continue

        seen_text.add(text)

        result = result.copy()
        result["rrf_score"] = scores[doc_id]

        final_results.append(result)

        if len(final_results) >= k:
            break

    return final_results


def fit_context(chunks, budget):
    encoding = tiktoken.get_encoding("cl100k_base")

    context_parts = []
    used_tokens = 0

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        filename = metadata.get(
            "filename",
            "Unknown source"
        )

        page = metadata.get("page")

        if page is not None:
            source = f"[Source: {filename}, Page {page}]"
        else:
            source = f"[Source: {filename}]"

        formatted_chunk = (
            f"{source}\n{chunk.get('text', '')}"
        )

        tokens = encoding.encode(formatted_chunk)

        remaining = budget - used_tokens

        if remaining <= 0:
            break

        if len(tokens) > remaining:
            if not context_parts:
                context_parts.append(
                    encoding.decode(tokens[:remaining])
                )
            break

        context_parts.append(formatted_chunk)
        used_tokens += len(tokens)

    return "\n\n".join(context_parts)


def answer(question, k=4):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    print("Retrieving relevant chunks...", flush=True)

    chunks = hybrid_retrieve(question, k=k)

    if not chunks:
        return FALLBACK_ANSWER

    print("\nHYBRID RETRIEVAL RESULTS")

    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        print(
            f"Rank {rank}: "
            f"{metadata.get('filename')} "
            f"(RRF: {chunk['rrf_score']:.4f})"
        )

    best_score = max(
        chunk["score"] for chunk in chunks
    )

    print(
        "Best vector similarity score:",
        round(best_score, 4),
        flush=True
    )

    if best_score < SIMILARITY_THRESHOLD:
        return FALLBACK_ANSWER

    context = fit_context(
        chunks,
        TOKEN_BUDGET
    )

    if not context.strip():
        return FALLBACK_ANSWER

    if not API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is missing from .env"
        )

    print("Sending request to NVIDIA...", flush=True)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
   
"You are a precise company policy assistant. "
"Answer using ONLY the retrieved context. "
"Never use outside knowledge or make assumptions. "

"Write short, complete, grammatically correct sentences. "
"Answer every part of the question. "
"Do not repeat the same fact. "
"Combine related facts into one concise answer. "

"For questions beginning with who, identify the person or role "
"and clearly state their responsibility. "

"Every factual statement must include a source citation. "
"Use this exact citation format: "
"[Source: filename.pdf, Page 1] for PDF documents. "
"[Source: filename.docx] for DOCX documents. "
"Copy filenames and page numbers exactly from the context. "
"Never invent filenames or page numbers. "

"If the context does not contain enough information, "
"respond exactly: I don't have information on that"

)
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        "max_tokens": 4096,
        "reasoning_budget": 1024,
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    generated_answer = (
        result["choices"][0]["message"]["content"]
    )

    if not generated_answer:
        raise RuntimeError(
            "NVIDIA returned an empty answer."
        )

    print("NVIDIA response received.", flush=True)

    return generated_answer


if __name__ == "__main__":

    question = "Who approves exceptions to company policies?"

    print("\nQUESTION:", question)
    print("\nANSWER:")
    print(answer(question, k=4))
