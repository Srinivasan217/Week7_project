# Week 7 – RAG Assessment

## Project Overview

## Technologies Used

## Project Structure

## Task 1 – RAG Pipeline

## Task 2 – Advanced Retrieval
## Cross-Encoder Reranking Results

- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Questions evaluated: 10
- Average rank before reranking: 2.5
- Average rank after reranking: 1.0
- Correct document ranked first after reranking: 10/10

Note: The current Qdrant collection contains 10 chunks,
so the evaluation used 10 candidates rather than the
required top 50.
## Task 3 – Citations and Hallucination Control

## Task 4 – Golden Dataset and RAGAS Evaluation

## Business Summary### Business Summary – RAG Pipeline Improvement

The RAG-based company policy assistant was evaluated using a golden dataset of 15 questions, including simple questions, multi-part questions, and unanswerable questions.

The initial evaluation identified retrieval issues caused by fixed-character chunking, which split policy statements into incomplete fragments and introduced overlapping content.

To improve retrieval quality, the document chunking strategy was changed to sentence-aware chunking with a target size of 700 characters. The Qdrant index was rebuilt using the improved chunks.

The evaluation showed improvements across all four RAGAS metrics:

- Faithfulness increased from 0.5000 to 0.5952.
- Answer relevancy increased from 0.5822 to 0.5993.
- Context precision increased from 0.7093 to 0.8000.
- Context recall increased from 0.7667 to 0.8000.

The improvement helped retrieve more complete and relevant policy information. However, faithfulness remains below the target of 0.85. Further improvements to retrieval thresholds, answer generation, and evaluation reliability are recommended.

The system also maintains fallback responses for questions that cannot be answered using the available company documents.
