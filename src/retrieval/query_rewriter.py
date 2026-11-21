
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.task2_questions import questions


class QueryRewriter:

    def __init__(self, rrf_constant=60):
        self.retriever = HybridRetriever()
        self.rrf_constant = rrf_constant

    def rewrite_query(self, question):
        return [
            question,
            f"What policy document covers: {question}",
            f"Which company document provides information about: {question}",
            f"Find the relevant company rule or guideline for: {question}"
        ]

    def retrieve(self, question, k=3):
        rewritten_queries = self.rewrite_query(question)

        scores = {}
        documents = {}

        for query in rewritten_queries:
            results = self.retriever.retrieve(query, k=10)

            for rank, result in enumerate(results, start=1):
                doc_id = result["id"]

                scores[doc_id] = scores.get(doc_id, 0) + (
                    1 / (self.rrf_constant + rank)
                )

                documents[doc_id] = result

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        final_results = []

        for doc_id in ranked_ids[:k]:
            result = documents[doc_id].copy()
            result["rrf_score"] = scores[doc_id]
            final_results.append(result)

        return final_results


def evaluate_query_rewriting():
    rewriter = QueryRewriter()

    print("\nQUERY REWRITING EVALUATION")
    print("=" * 60)

    for item in questions:
        question = item["question"]
        expected_file = item["filename"]

        start = time.time()
        results = rewriter.retrieve(question, k=3)
        latency = time.time() - start

        hit = any(
            result.get("metadata", {}).get("filename") == expected_file
            or result.get("filename") == expected_file
            for result in results
        )

        print(f"\nQuestion: {question}")
        print("Expected:", expected_file)
        print("Hit@3:", hit)
        print("Latency:", round(latency, 3), "seconds")


if __name__ == "__main__":
    evaluate_query_rewriting()
