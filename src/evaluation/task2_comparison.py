
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))

from bm25_retriever import BM25Retriever
from retriever import retrieve_chunks
from hybrid_retriever import HybridRetriever
from src.evaluation.task2_questions import questions


def get_filename(result):
    return (
        result.get("filename")
        or result.get("metadata", {}).get("filename")
    )


def evaluate():
    bm25 = BM25Retriever()
    hybrid = HybridRetriever()

    results = {
        "exact": {"bm25": 0, "vector": 0, "hybrid": 0},
        "semantic": {"bm25": 0, "vector": 0, "hybrid": 0}
    }

    for item in questions:
        question = item["question"]
        question_type = item["type"]
        expected_file = item["filename"]

        bm25_results = bm25.retrieve(question, k=3)
        vector_results = retrieve_chunks(question, k=3)
        hybrid_results = hybrid.retrieve(question, k=3)

        bm25_hit = expected_file in [
            get_filename(r) for r in bm25_results
        ]

        vector_hit = expected_file in [
            get_filename(r) for r in vector_results
        ]

        hybrid_hit = expected_file in [
            get_filename(r) for r in hybrid_results
        ]

        if bm25_hit:
            results[question_type]["bm25"] += 1

        if vector_hit:
            results[question_type]["vector"] += 1

        if hybrid_hit:
            results[question_type]["hybrid"] += 1

        print("\nQUESTION:", question)
        print("EXPECTED:", expected_file)
        print("BM25 HIT@3:", bm25_hit)
        print("VECTOR HIT@3:", vector_hit)
        print("HYBRID HIT@3:", hybrid_hit)

    print("\n========== FINAL HIT@3 RESULTS ==========")

    for question_type in ["exact", "semantic"]:
        print(f"\n{question_type.upper()} QUESTIONS")

        for method in ["bm25", "vector", "hybrid"]:
            print(
                method.upper(),
                results[question_type][method],
                "/ 6"
            )
    print("\nOVERALL RESULTS")
    for method in ["bm25", "vector", "hybrid"]:
        total = sum(results[t][method] for t in results)
        print(method.upper(),total, "/12")



if __name__ == "__main__":
    evaluate()
