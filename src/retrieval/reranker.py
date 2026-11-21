
from sentence_transformers import CrossEncoder

from src.retrieval.retriever import retrieve_chunks
from src.evaluation.task2_questions import questions


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 50
FINAL_K = 5


def find_expected_rank(results, expected):
    expected = expected.lower()

    for rank, item in enumerate(results, start=1):
        chunk = item[0] if isinstance(item, tuple) else item

        if expected in chunk["text"].lower():
            return rank

    return None


def evaluate_reranking():
    model = CrossEncoder(MODEL_NAME)

    print("\nEXACT CHUNK RERANKING EVALUATION")
    print("=" * 65)

    for item in questions[:10]:
        question = item["question"]
        expected = item["expected"]

        candidates = retrieve_chunks(question, k=CANDIDATE_K)

        before_rank = find_expected_rank(candidates, expected)

        pairs = [
            (question, chunk["text"])
            for chunk in candidates
        ]

        scores = model.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda result: result[1],
            reverse=True
        )

        after_rank = find_expected_rank(reranked, expected)

        print(f"\nQuestion: {question}")
        print(f"Expected answer: {expected}")
        print(f"Candidates retrieved: {len(candidates)}")
        print(f"Correct chunk rank before: {before_rank}")
        print(f"Correct chunk rank after:  {after_rank}")

        print(f"\nTop {FINAL_K} chunks after reranking:")

        for rank, (chunk, score) in enumerate(
            reranked[:FINAL_K],
            start=1
        ):
            filename = chunk["metadata"].get("filename", "Unknown")

            print(
                f"{rank}. {filename} | "
                f"Score: {float(score):.4f}"
            )


if __name__ == "__main__":
    evaluate_reranking()
