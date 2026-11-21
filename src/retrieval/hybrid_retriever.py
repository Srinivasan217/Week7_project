from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT / "src"/"retrieval"))

from bm25_retriever import BM25Retriever
from src.retrieval.retriever import retrieve_chunks

class HybridRetriever:
    def __init__(self,k=3, rrf_constant=60):
        self.k = k
        self.rrf_constant = rrf_constant
        self.bm25 = BM25Retriever()

    def retrieve(self, question, k=None):
        if k is None:
            k = self.k    

        bm25_results = self.bm25.retrieve(question, k=k)
        vector_results = retrieve_chunks(question, k=k)

        scores = {}
        documents = {}

        # Add BM25 rankings
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (self.rrf_constant + rank)
            )

            documents[doc_id] = result

        # Add Vector rankings
        for rank, result in enumerate(vector_results, start=1):
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


if __name__ == "__main__":
    retriever = HybridRetriever()

    question = "Which policy explains how workers should manage their time off?"

    results = retriever.retrieve(question, k=3)

    print("\nHYBRID RETRIEVAL RESULTS")

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        filename = result.get("filename") or metadata.get("filename")

        print(f"\nRank: {rank}")
        print("File:", filename)
        print("RRF Score:", round(result["rrf_score"], 6))
