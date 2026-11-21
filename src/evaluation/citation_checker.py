
import re
import json
from pathlib import Path

from src.retrieval.retriever import retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_citations(text):
    pattern = r"\[Source:\s*([^,\]]+)(?:,\s*Page\s+(\d+))?\]"
    matches = re.findall(pattern, text)

    citations = []

    for filename, page in matches:
        citations.append({
            "filename": filename.strip(),
            "page": int(page) if page else None
        })

    return citations


def check_citations(answer_text, chunks):
    citations = parse_citations(answer_text)

    valid_sources = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        filename = metadata.get("filename")
        page = metadata.get("page")

        valid_sources.add((filename, page))

    invalid_citations = []

    for citation in citations:
        filename = citation["filename"]
        page = citation["page"]

        if page is None:
            matching_source = any(
                source_filename == filename
                and source_page is None
                for source_filename, source_page in valid_sources
            )
        else:
            matching_source = (filename, page) in valid_sources

        if not matching_source:
            invalid_citations.append(citation)

    return citations, invalid_citations


if __name__ == "__main__":

    results_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "task3_prompt_results_v2.json"
    )

    with open(results_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    # Select answers from the strongest citation prompt.
    selected_results = [
        item for item in results
        if item["prompt"] == "Context + Citation + Fallback"
        and item["answerable"] is True
    ]

    print("\nCITATION VALIDATION")
    print("=" * 60)
    print("Answerable questions:", len(selected_results))

    total_citations = 0
    valid_citations = 0
    invalid_citations = 0
    missing_citations = 0

    for index, item in enumerate(selected_results, start=1):
        question = item["question"]
        generated_answer = item["answer"]

        chunks = retrieve_chunks(question, k=4)

        citations, invalid = check_citations(
            generated_answer,
            chunks
        )

        total_citations += len(citations)
        invalid_citations += len(invalid)
        valid_citations += len(citations) - len(invalid)

        if not citations:
            missing_citations += 1

        print(f"\nQUESTION {index}: {question}")
        print("ANSWER:", generated_answer)
        print("CITATIONS:", citations)
        print("INVALID:", invalid)

        if not citations:
            print("STATUS: Missing citation")
        elif invalid:
            print("STATUS: Invalid citation")
        else:
            print("STATUS: Valid citations")

    print("\n" + "=" * 60)
    print("FINAL CITATION SUMMARY")
    print("=" * 60)
    print("Questions checked:", len(selected_results))
    print("Questions missing citations:", missing_citations)
    print("Total citations:", total_citations)
    print("Valid citations:", valid_citations)
    print("Invalid citations:", invalid_citations)

    if (
        len(selected_results) == 7
        and missing_citations == 0
        and invalid_citations == 0
    ):
        print("\nRESULT: All 7 answerable questions passed citation validation.")
    else:
        print("\nRESULT: Some citation issues need attention.")
