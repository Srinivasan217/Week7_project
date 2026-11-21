
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
    Remove duplicate text while preserving source metadata.
    """

    vector_results = retrieve_chunks(
        question,
        k=RETRIEVAL_CANDIDATES
    )

    bm25_results = bm25_retriever.retrieve(
        question,
        k=RETRIEVAL_CANDIDATES
    )

    scores = {}
    documents = {}

    # Vector ranking
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (60 + rank)
        )

        documents[doc_id] = result

    # BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (60 + rank)
        )

        # Normalize BM25 result structure
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

        # Avoid identical content from different documents
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
    "Do not use outside knowledge or make assumptions. "

    "Always answer in complete, grammatically correct sentences. "
    "Never return only a person, role, noun phrase, or sentence fragment. "
    "For questions beginning with who, identify the person or role "
    "and clearly state their responsibility. "
    "For example, answer 'Who approves exceptions?' with "
    "'Exceptions require approval from the authorized manager.' "

    "Give a short, direct answer addressing exactly what was asked. "
    "Do not combine unrelated policies or introduce unsupported details. "

    "Every factual statement must include a citation. "
    "Copy citations exactly as they appear in the retrieved context. "
    "Never invent filenames or page numbers. "
    "Preserve page numbers when provided. "
    "If no page number is provided, cite only the filename. "

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
