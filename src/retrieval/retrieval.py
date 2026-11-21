import os
from pathlib import Path
import sys

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.retrieval.retriever import retrieve_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0,str(PROJECT_ROOT / "src"/"retrieval"))
from src.retrieval.retrieval import retrieve_chunks
from src.evaluation.task2_questions import questions


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def evaluate_reranking():
    print("Loading Cross-Encoder model...")
    model = CrossEncoder(MODEL_NAME)

    total_before = 0
    total_after = 0

    for item in questions[:10]:
        question = item["question"]
        expected_file = item["filename"]

        # Retrieve up to 50 candidate chunks
        candidates = retrieve_chunks(question, k=50)

        if not candidates:
            print("\nQUESTION:", question)
            print("No candidates retrieved.")
            continue

        # Find correct document's rank before reranking
        before_rank = None

        for rank, result in enumerate(candidates, start=1):
            filename = result["metadata"]["filename"]

            if filename == expected_file:
                before_rank = rank
                break

        # Prepare question and document pairs
        pairs = [
            (question, result["text"])
            for result in candidates
        ]

        # Cross-Encoder scoring
        scores = model.predict(pairs)

        # Sort candidates by Cross-Encoder score
        reranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Find correct document's rank after reranking
        after_rank = None

        for rank, (result, score) in enumerate(reranked, start=1):
            filename = result["metadata"]["filename"]

            if filename == expected_file:
                after_rank = rank
                break

        if before_rank is not None and before_rank <= 5:
            total_before += 1

        if after_rank is not None and after_rank <= 5:
            total_after += 1

        print("\nQUESTION:", question)
        print("EXPECTED FILE:", expected_file)
        print("CANDIDATES RETRIEVED:", len(candidates))
        print("RANK BEFORE RERANKING:", before_rank)
        print("RANK AFTER RERANKING:", after_rank)

        print("\nTOP 5 AFTER RERANKING:")

        for rank, (result, score) in enumerate(reranked[:5], start=1):
            filename = result["metadata"]["filename"]

            print(
                f"{rank}. {filename} "
                f"(score: {float(score):.4f})"
            )

    print("\n========== RERANKING SUMMARY ==========")
    print("Questions evaluated:", len(questions[:10]))
    print("Correct documents in top 5 before:", total_before)
    print("Correct documents in top 5 after:", total_after)


if __name__ == "__main__":
    evaluate_reranking()
