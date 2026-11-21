
import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.evaluation.task2_questions import questions


def evaluate_method(name, retriever):
    hits = 0
    total_latency = 0

    print(f"\n{name} RESULTS")
    print("=" * 50)

    for item in questions:
        question = item["question"]
        expected_file = item["filename"]

        start = time.perf_counter()
        results = retriever.retrieve(question, k=3)
        latency = time.perf_counter() - start

        total_latency += latency

        hit = any(
            result.get("metadata", {}).get("filename") == expected_file
            or result.get("filename") == expected_file
            for result in results
        )

        if hit:
            hits += 1

        print(f"Question: {question}")
        print(f"Expected: {expected_file}")
        print(f"Hit@3: {hit}")
        print(f"Latency: {latency:.4f} seconds")

    average_latency = total_latency / len(questions)

    return {
        "method": name,
        "hits": hits,
        "accuracy": hits / len(questions) * 100,
        "average_latency": average_latency
    }


def main():
    hybrid = HybridRetriever()
    query_rewriter = QueryRewriter()

    hybrid_result = evaluate_method("Hybrid Retrieval", hybrid)
    rewrite_result = evaluate_method("Query Rewriting", query_rewriter)

    print("\nFINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<22} {'Hit@3':<12} {'Accuracy':<12} {'Avg Latency'}")

    for result in [hybrid_result, rewrite_result]:
        print(
            f"{result['method']:<22} "
            f"{result['hits']}/12{'':<7} "
            f"{result['accuracy']:.2f}%{'':<6} "
            f"{result['average_latency']:.4f}s"
        )


if __name__ == "__main__":
    main()
