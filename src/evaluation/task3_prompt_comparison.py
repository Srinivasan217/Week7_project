
import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src.generation.answerer import retrieve_chunks, fit_context


API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


QUESTIONS = [
    # 7 answerable questions
    {
        "question": "Who is responsible for communicating company guidelines?",
        "answerable": True,
    },
    {
        "question": "Who should employees contact for clarification?",
        "answerable": True,
    },
    {
        "question": "Who must approve exceptions?",
        "answerable": True,
    },
    {
        "question": "How often should the documents be reviewed?",
        "answerable": True,
    },
    {
        "question": "What should employees follow?",
        "answerable": True,
    },
    {
        "question": "What is the purpose of the payroll processing policy?",
        "answerable": True,
    },
    {
        "question": "What is the title of the annual leave document?",
        "answerable": True,
    },

    # 3 unanswerable questions
    {
        "question": "Who is the company CEO?",
        "answerable": False,
    },
    {
        "question": "What is the company's street address?",
        "answerable": False,
    },
    {
        "question": "How do I bake a chocolate cake?",
        "answerable": False,
    },
]


PROMPTS = {
    "Basic": (
        "Answer the user's question using the information provided "
        "in the context. Give a clear and helpful answer."
    ),

    "Context Only": (
        "Answer ONLY using the context provided. "
        "Do not use outside knowledge. "
        "If the context does not contain the answer, "
        "say: I don't have information on that."
    ),

    "Context + Citation + Fallback": (
        "Answer ONLY using the context provided. "
        "Do not use outside knowledge. "
        "If the context does not contain the answer, "
        "say: I don't have information on that. "
        "Cite every factual statement using the exact source format "
        "shown in the context, such as [Source: finance_01.pdf, Page 1]. "
        "Never invent or modify source filenames or page numbers."
    ),
}



def is_abstention(answer):
    answer = answer.lower()
    answer = answer.replace("’", "'").replace("‘", "'")
    abstention_phrases = [
        "i don't have information",
        "i do not have information",
        "not mentioned in the context",
        "not mentioned in the documents",
        "does not contain the answer",
        "doesn't contain the answer",
        "does not contain any information",
        "don't contain any information",
        "does not contain any details",
        "doesn't contain any details",
        "context does not provide",
        "context doesn't provide",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "unable to determine",
        "cannot determine",
        "can't determine",
        "not provided in the context",
        "not available in the provided information",
        "documents do not mention",
        "documents don't mention",
        "there is no information",
        "not enough information",
        "not answerable",
        "not available in the excerpts",
        "don't contain any information about the company's CEO",
    ]

    return any(
        phrase in answer
        for phrase in abstention_phrases
    )


def main():

    if not API_KEY:
        raise ValueError("NVIDIA_API_KEY is missing from .env")
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=90,
        max_retries=2,
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "task3_prompt_results_v2.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
       with open(output_path, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
      results = []
    # Recalculate previous evaluation flags using the updated detector.
    for item in results:
        abstained = is_abstention(item["answer"])

        item["abstained"] = abstained
        item["potential_hallucination"] = (
            not item["answerable"] and not abstained
       )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    completed = {
        (item["question"], item["prompt"])
        for item in results
     }

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=90,
        max_retries=2,
     )

    print("Provider: NVIDIA")
    print("Model:", MODEL)
    print("Previously completed:", len(completed))

    for index, item in enumerate(QUESTIONS, start=1):

        question = item["question"]

        print("\n" + "=" * 65)
        print(f"QUESTION {index}: {question}")
        print("Answerable:", item["answerable"])

        chunks = retrieve_chunks(question, k=4)
        context = fit_context(chunks, 3000)

        for prompt_name, system_prompt in PROMPTS.items():

            if (question, prompt_name) in completed:
                print(prompt_name, "- Already completed. Skipping.")
                continue

            print(f"\n--- {prompt_name} ---")

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context}\n\n"
                                f"Question: {question}"
                            ),
                        },
                    ],
                    temperature=0,
                )

                generated_answer = (
                    response.choices[0].message.content or ""
                )

                print(generated_answer)

                abstained = is_abstention(generated_answer)

                potential_hallucination = (
                    not item["answerable"] and not abstained
                )

                result = {
                    "question": question,
                    "answerable": item["answerable"],
                    "prompt": prompt_name,
                    "answer": generated_answer,
                    "abstained": abstained,
                    "potential_hallucination": potential_hallucination,
                    "retrieved_sources": [
                        chunk.get("metadata", {}).get("filename")
                        for chunk in chunks
                    ],
                }

                results.append(result)

                with open(
                    output_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        results,
                        file,
                        indent=4,
                        ensure_ascii=False,
                    )

                completed.add((question, prompt_name))
                print("Abstained:", abstained)
                print("Potential hallucination:", potential_hallucination)
                print("Saved successfully.")

                time.sleep(5)

            except Exception as error:
                print("\nRequest failed:", error)
                print("Progress has been saved.")
                print("Run again to continue.")
                return

    print("\n" + "=" * 65)
    print("PROMPT COMPARISON COMPLETED")
    print("Total questions:", len(QUESTIONS))
    print("Answerable:", sum(q["answerable"] for q in QUESTIONS))
    print("Unanswerable:", sum(not q["answerable"] for q in QUESTIONS))
    print("Prompt versions:", len(PROMPTS))
    print("Total responses:", len(results))

    print("\nEvaluation results:")

    for prompt_name in PROMPTS:

        prompt_results = [
            item for item in results
            if item["prompt"] == prompt_name
        ]

        hallucinations = sum(
            item.get("potential_hallucination", False)
            for item in prompt_results
            if not item["answerable"]
        )

        abstentions = sum(
            item.get("abstained", False)
            for item in prompt_results
            if not item["answerable"]
        )

        print(f"\n{prompt_name}")
        print("Unanswerable questions:", len([
            item for item in prompt_results
            if not item["answerable"]
        ]))
        print("Abstentions:", abstentions)
        print("Potential hallucinations:", hallucinations)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
