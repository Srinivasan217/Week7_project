import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generation"))

from answerer import answer

questions = [
    ("ANSWERABLE", "Who is responsible for communicating company guidelines?"),
    ("ANSWERABLE", "Who should employees contact for clarification?"),
    ("ANSWERABLE", "Who must approve exceptions to the policy?"),
    ("ANSWERABLE", "What does the policy say about reviewing the document annually?"),
    ("ANSWERABLE", "What procedures must employees follow?"),
    ("ANSWERABLE", "What is the purpose of the payroll processing policy?"),
    ("UNANSWERABLE", "What is the name of the company's CEO?"),
    ("UNANSWERABLE", "What is the company's office street address?"),
    ("UNANSWERABLE", "What is the recipe for making chocolate cake?"),
    ("UNANSWERABLE", "Who won the latest cricket world cup?")
]

if __name__ == "__main__":
    for index, (question_type, question) in enumerate(questions, start=1):
        print("\n" + "=" * 60)
        print(f"QUESTION {index} - {question_type}")
        print("Question:", question)

        try:
            result = answer(question, k=4)
            print("Answer:", result)
        except Exception as error:
            print("ERROR:", error)

