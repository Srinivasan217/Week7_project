
import json
from pathlib import Path

from src.retrieval.retriever import retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REVIEW_QUESTIONS = [
    {
        "question": "Who is responsible for communicating company guidelines?",
        "answer": "Managers are responsible for communicating company guidelines."
    },
    {
        "question": "Who should employees contact for clarification?",
        "answer": "Employees should contact their department manager for clarification."
    },
    {
        "question": "Who approves policy exceptions?",
        "answer": "Exceptions require approval from the authorized manager."
    },
    {
        "question": "How often should company documents be reviewed?",
        "answer": "Company documents should be reviewed annually."
    },
    {
        "question": "What procedures must employees follow?",
        "answer": "Employees must follow approved company procedures."
    }
]


def main():
    results = []

    for index, item in enumerate(REVIEW_QUESTIONS, start=1):
        question = item["question"]
        candidate_answer = item["answer"]

        print("\n" + "=" * 65)
        print(f"REVIEW {index}")
        print("QUESTION:", question)
        print("ANSWER:", candidate_answer)

        chunks = retrieve_chunks(question, k=4)

        print("\nRETRIEVED CONTEXT:")

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            print("\nSource:", metadata.get("filename"))
            print("Page:", metadata.get("page"))
            print("Evidence:", chunk.get("text", ""))

        print("\nMANUAL CHECK")
        print("F = Faithful (supported by context)")
        print("N = Not faithful (unsupported by context)")

        while True:
            decision = input("Is this answer faithful? (F/N): ").strip().upper()

            if decision in ("F", "N"):
                break

            print("Please enter F or N.")

        note = input("Enter a short reason: ").strip()

        results.append({
            "question": question,
            "answer": candidate_answer,
            "faithful": decision == "F",
            "review_note": note
        })

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "task3_manual_faithfulness.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    faithful_count = sum(item["faithful"] for item in results)

    print("\n" + "=" * 65)
    print("MANUAL FAITHFULNESS REVIEW COMPLETED")
    print("Total answers reviewed:", len(results))
    print("Faithful answers:", faithful_count)
    print("Not faithful:", len(results) - faithful_count)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
