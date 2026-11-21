import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Retriever:

    def __init__(self):
        self.documents = []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.documents.append(json.loads(line))

        self.tokenized_documents = [
            tokenize(document["text"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print(f"BM25 loaded {len(self.documents)} document chunks.")

    def retrieve(self, question, k=4):
        query_tokens = tokenize(question)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:k]:
            document = self.documents[index]

            results.append({
                "id": document["id"],
                "text": document["text"],
                "filename": document["filename"],
                "department": document["department"],
                "page": document["page"],
                "score": float(scores[index])
            })

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    question = "Who is responsible for communicating company guidelines?"

    results = retriever.retrieve(question, k=4)

    print("\nQuestion:", question)
    print("\nBM25 Search Results:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"File: {result['filename']}")
        print(f"Department: {result['department']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:200]}")
