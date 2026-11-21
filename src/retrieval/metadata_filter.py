import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT/"data"/ "processed"/ "chunks.jsonl"


class MetadataFilter:

    def __init__(self):
        self.chunks = []

        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.chunks.append(json.loads(line))

    def filter_documents(self, department=None, year=None):
        results = self.chunks

        if department:
            results = [
                chunk for chunk in results
                if chunk.get("department", "").lower() == department.lower()
            ]

        if year:
            results = [
                chunk for chunk in results
                if chunk.get("year") == year
            ]

        return results


def main():
    metadata_filter = MetadataFilter()

    # Department filtering
    finance_docs = metadata_filter.filter_documents(
        department="finance"
    )

    print("\nFINANCE DEPARTMENT DOCUMENTS")
    for doc in finance_docs:
        print(doc["filename"])

    # Sample year metadata for demonstration only
    sample_years = {
        "finance_01.pdf": 2023,
        "finance_02.docx": 2025,
    }

    for doc in metadata_filter.chunks:
        doc["year"] = sample_years.get(doc["filename"], 2025)

    # Year filtering
    current_docs = [
        doc for doc in metadata_filter.chunks
        if doc["year"] >= 2024
    ]

    excluded_docs = [
        doc for doc in metadata_filter.chunks
        if doc["year"] < 2024
    ]

    print("\nYEAR FILTERING")
    print("Included documents:")
    for doc in current_docs:
        print(doc["filename"], "Year:", doc["year"])

    print("\nExcluded outdated documents:")
    for doc in excluded_docs:
        print(doc["filename"], "Year:", doc["year"])


if __name__ == "__main__":
    main()