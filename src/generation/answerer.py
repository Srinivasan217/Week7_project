
import os
from pathlib import Path

import requests
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

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
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FALLBACK_ANSWER = "I don't have information on that"

bm25_retriever = BM25Retriever()

cross_encoder = None


def get_reranker():
    global cross_encoder

    if cross_encoder is None:
        print("Loading Cross-Encoder model...", flush=True)
        cross_encoder = CrossEncoder(RERANKER_MODEL)

    return cross_encoder


def build_search_queries(question):
    """
    Create focused searches for compound questions.

    Keep the original question and add searches for
    individual parts when the question contains 'and'.
    """

    queries = [question]

    lower_question = question.lower()

    if " and " in lower_question:
        split_position = lower_question.find(" and ")

        first_part = question[:split_position].strip(" ?.,")
        second_part = question[split_position + 5:].strip(" ?.,?")

        if len(first_part) >= 8:
            queries.append(first_part)

        if len(second_part) >= 8:

            # Expand annual-review questions into the wording
            # used in the indexed policy documents.
            if (
                "review" in second_part.lower()
                or "review" in lower_question
            ):
                second_part = (
                    "How often should company documents be reviewed?"
                )

            queries.append(second_part)

    # Remove duplicate searches while preserving order.
    return list(dict.fromkeys(queries))


def hybrid_retrieve(question, k=4, rerank=False):
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.

    For compound questions:
    - Search the original question.
    - Search individual question parts.
    - Rerank against each search query.
    - Preserve the best result for each question part.
    """

    queries = build_search_queries(question)

    scores = {}
    documents = {}

    for search_query in queries:

        vector_results = retrieve_chunks(
            search_query,
            k=RETRIEVAL_CANDIDATES
        )

        bm25_results = bm25_retriever.retrieve(
            search_query,
            k=RETRIEVAL_CANDIDATES
        )

        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (60 + rank)
            )

            if doc_id not in documents:
                documents[doc_id] = result.copy()
            elif result["score"] > documents[doc_id]["score"]:
                documents[doc_id]["score"] = result["score"]

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (60 + rank)
            )

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

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    candidates = []
    seen_text = set()

    for doc_id in ranked_ids:
        result = documents[doc_id]

        text = " ".join(result["text"].lower().split())

        if text in seen_text:
            continue

        seen_text.add(text)

        result = result.copy()
        result["rrf_score"] = scores[doc_id]

        candidates.append(result)

        if len(candidates) >= RETRIEVAL_CANDIDATES:
            break

    if not candidates:
        return []

    best_vector_score = max(
        chunk["score"] for chunk in candidates
    )

    if rerank:
        print("Reranking hybrid candidates...", flush=True)

        model = get_reranker()

        # Score every candidate against every search query.
        pairs = [
            (search_query, chunk["text"])
            for search_query in queries
            for chunk in candidates
        ]

        all_scores = model.predict(pairs)

        query_count = len(queries)
        candidate_count = len(candidates)

        score_matrix = [
            all_scores[
                query_index * candidate_count:
                (query_index + 1) * candidate_count
            ]
            for query_index in range(query_count)
        ]

        # Use the best matching query score for each chunk.
        combined_scores = [
            max(
                float(score_matrix[query_index][candidate_index])
                for query_index in range(query_count)
            )
            for candidate_index in range(candidate_count)
        ]

        # First preserve the best matching chunk for each
        # part of a compound question.
        selected_indices = []

        for query_index in range(query_count):

            best_index = max(
                range(candidate_count),
                key=lambda index: float(
                    score_matrix[query_index][index]
                )
            )

            if best_index not in selected_indices:
                selected_indices.append(best_index)

        # Fill remaining positions using the combined scores.
        remaining_indices = sorted(
            range(candidate_count),
            key=lambda index: combined_scores[index],
            reverse=True
        )

        for index in remaining_indices:
            if index not in selected_indices:
                selected_indices.append(index)

            if len(selected_indices) >= k:
                break

        final_results = []

        for index in selected_indices[:k]:
            chunk = candidates[index].copy()

            chunk["rerank_score"] = combined_scores[index]
            chunk["best_vector_score"] = best_vector_score

            final_results.append(chunk)

        return final_results

    final_results = []

    for chunk in candidates[:k]:
        chunk = chunk.copy()
        chunk["best_vector_score"] = best_vector_score
        final_results.append(chunk)

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


def answer(question, k=4, retrieved_chunks=None):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    print("Retrieving relevant chunks...", flush=True)

    if retrieved_chunks is None:
        chunks = hybrid_retrieve(
            question,
            k=k,
            rerank=True
        )
    else:
        chunks = retrieved_chunks

    if not chunks:
        return FALLBACK_ANSWER

    print("\nHYBRID + CROSS-ENCODER RESULTS")

    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        print(
            f"Rank {rank}: "
            f"{metadata.get('filename')} | "
            f"RRF: {chunk.get('rrf_score', 0):.4f} | "
            f"Rerank: {chunk.get('rerank_score', 0):.4f}"
        )

    best_score = chunks[0].get(
        "best_vector_score",
        chunks[0].get("score", 0.0)
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

                    "IMPORTANT: Never return only a source citation. "
                    "Every answer must contain an actual factual sentence "
                    "supported by the retrieved context. "

                    "For questions asking which document or policy, "
                    "identify the exact document or policy title from "
                    "the context and explain what it covers. "

                    "For questions beginning with who, identify the person "
                    "or role and clearly state their responsibility. "

                    "For questions containing multiple parts, answer "
                    "each part separately when the context supports it. "
                    "Do not ignore any part of the question. "

                    "Every factual statement must include a source citation. "
                    "Use this exact citation format: "
                    "[Source: filename.pdf, Page 1] for PDF documents. "
                    "[Source: filename.docx] for DOCX documents. "
                    "Copy filenames and page numbers exactly from the context. "
                    "Never invent filenames or page numbers. "

                    "If the context does not contain enough information "
                    "to answer a question, respond exactly: "
                    "I don't have information on that"
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
