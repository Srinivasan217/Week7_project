
import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.generation.answerer import answer, hybrid_retrieve


GOLDEN_FILE = PROJECT_ROOT / "evaluation" / "golden.jsonl"

# Separate filenames preserve all baseline results.
DATASET_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "task4_ragas_dataset_reranked_20261121.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "task4_ragas_results_reranked_20261121.json"
)

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

RETRIEVAL_K = 4

FALLBACK_ANSWER = "I don't have information on that"


def format_context(chunk):
    metadata = chunk.get("metadata", {})

    filename = metadata.get("filename", "Unknown source")
    page = metadata.get("page")

    if page is not None:
        source = f"[Source: {filename}, Page {page}]"
    else:
        source = f"[Source: {filename}]"

    return f"{source}\n{chunk.get('text', '')}"


def is_abstention(response):
    normalized = " ".join(response.lower().split())
    normalized = normalized.rstrip(".! ")

    expected = FALLBACK_ANSWER.lower()

    return normalized == expected


def prepare_dataset():
    print("Loading golden dataset...", flush=True)

    with open(GOLDEN_FILE, "r", encoding="utf-8") as file:
        golden_data = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    results = []

    for index, item in enumerate(golden_data, start=1):
        question = item["question"]

        print(
            f"\nGenerating answer {index}/{len(golden_data)}",
            flush=True
        )
        print("Question:", question, flush=True)

        # Retrieve once, using hybrid retrieval and Cross-Encoder.
        chunks = hybrid_retrieve(
            question,
            k=RETRIEVAL_K,
            rerank=True
        )

        # Pass the same chunks to the answer generator.
        generated_answer = answer(
            question,
            k=RETRIEVAL_K,
            retrieved_chunks=chunks
        )

        results.append({
            "question": question,
            "answer": generated_answer,
            "contexts": [
                format_context(chunk)
                for chunk in chunks
            ],
            "ground_truth": item["ground_truth"],
            "source": item["source"],
            "page": item["page"],
            "answerable": bool(item["source"]),
            "abstained": is_abstention(generated_answer)
        })

        # Save after every question.
        DATASET_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(DATASET_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print("Answer saved.", flush=True)

    print("\nFinal dataset saved:", DATASET_FILE, flush=True)

    return results


async def run_evaluation():

    if not API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is missing from .env"
        )

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = prepare_dataset()

    print("\nConnecting to NVIDIA for RAGAS...", flush=True)

    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=120,
        max_retries=0
    )

    ragas_llm = llm_factory(
        MODEL_NAME,
        provider="openai",
        client=client
    )

    ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings
        ),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm)
    ]

    all_results = []

    print(
        "\nStarting reranked RAGAS evaluation...",
        flush=True
    )

    for metric in metrics:

        print(
            f"\nEvaluating {metric.name}...",
            flush=True
        )

        scores = []

        for index, item in enumerate(dataset, start=1):

            print(
                f"Question {index}/{len(dataset)}",
                flush=True
            )

            if metric.name == "faithfulness":
                arguments = {
                    "user_input": item["question"],
                    "response": item["answer"],
                    "retrieved_contexts": item["contexts"]
                }

            elif metric.name == "answer_relevancy":
                arguments = {
                    "user_input": item["question"],
                    "response": item["answer"]
                }

            elif metric.name == "context_precision":
                arguments = {
                    "user_input": item["question"],
                    "reference": item["ground_truth"],
                    "retrieved_contexts": item["contexts"]
                }

            elif metric.name == "context_recall":
                arguments = {
                    "user_input": item["question"],
                    "reference": item["ground_truth"],
                    "retrieved_contexts": item["contexts"]
                }

            else:
                scores.append(None)
                continue

            try:
                result = await metric.ascore(**arguments)
                score = result.value
                scores.append(score)

                print("Score:", score, flush=True)

            except Exception as error:
                print(
                    "Evaluation error:",
                    repr(error),
                    flush=True
                )
                scores.append(None)

        valid_scores = [
            score for score in scores
            if score is not None
        ]

        answerable_scores = [
            score
            for score, item in zip(scores, dataset)
            if item["answerable"] and score is not None
        ]

        average_all = (
            sum(valid_scores) / len(valid_scores)
            if valid_scores else None
        )

        average_answerable = (
            sum(answerable_scores) / len(answerable_scores)
            if answerable_scores else None
        )

        metric_result = {
            "metric": metric.name,
            "average_all_questions": average_all,
            "average_answerable_questions": average_answerable,
            "valid_samples": len(valid_scores),
            "total_samples": len(scores),
            "answerable_valid_samples": len(answerable_scores),
            "individual_scores": scores
        }

        all_results.append(metric_result)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=4)

        print("All-question average:", average_all, flush=True)
        print("Answerable-only average:", average_answerable, flush=True)

    # Evaluate abstention accuracy separately.
    unanswerable_items = [
        item for item in dataset
        if not item["answerable"]
    ]

    correct_abstentions = sum(
        1 for item in unanswerable_items
        if item["abstained"]
    )

    abstention_accuracy = (
        correct_abstentions / len(unanswerable_items)
        if unanswerable_items else None
    )

    final_summary = {
        "evaluation_type": "hybrid_retrieval_with_cross_encoder_reranking",
        "total_questions": len(dataset),
        "answerable_questions": sum(
            1 for item in dataset if item["answerable"]
        ),
        "unanswerable_questions": len(unanswerable_items),
        "correct_abstentions": correct_abstentions,
        "abstention_accuracy": abstention_accuracy,
        "metrics": all_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(final_summary, file, indent=4)

    print("\n========== RERANKED RAGAS RESULTS ==========")

    for result in all_results:
        print(
            result["metric"],
            "| All questions:",
            result["average_all_questions"],
            "| Answerable only:",
            result["average_answerable_questions"]
        )

    print(
        "\nAbstention accuracy:",
        f"{correct_abstentions}/{len(unanswerable_items)}",
        "=",
        abstention_accuracy
    )

    print("\nDataset:", DATASET_FILE)
    print("Results:", OUTPUT_FILE, flush=True)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
