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
# Week 7 – RAG Assessment: Business Summary

The RAG-based company policy assistant was developed to help employees retrieve accurate information from internal company documents. The system supports PDF and DOCX documents, document chunking, embedding generation, Qdrant vector storage, and hybrid retrieval using vector search and BM25 with Reciprocal Rank Fusion.

Advanced retrieval features include cross-encoder reranking, query rewriting, and metadata filtering. The answer generation module uses an NVIDIA-hosted language model and is instructed to answer only from retrieved document context. Source citations are included in generated answers, and unsupported questions receive a predefined fallback response.

A golden evaluation dataset containing 15 questions was created, including simple questions, multi-part questions, and unanswerable questions. RAGAS evaluation produced the following results:

- Faithfulness: 0.6154
- Answer Relevancy: 0.5889
- Context Precision: 0.6704
- Context Recall: 0.7333

The evaluation identified opportunities to improve answer relevance, retrieval quality, and automated evaluation reliability. Manual inspection confirmed that several answers were correctly supported by retrieved documents, while unsupported questions were correctly handled through abstention. Some RAGAS judge evaluations returned incomplete outputs.

Overall, the project demonstrates a functional document-grounded question-answering system with hybrid retrieval, citations, and hallucination-control mechanisms. Further improvements are recommended before deploying the system in a production environment.
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
# Business Summary – Week 7 RAG Assessment

The Week 7 assessment developed and evaluated a Retrieval-Augmented Generation (RAG) system for answering questions from company policy documents.

The system supports document ingestion, text chunking, embedding generation, vector storage, semantic retrieval, BM25 keyword retrieval, hybrid retrieval, and cross-encoder reranking. It also includes source citations, hallucination controls, context-budget handling, and a protected FastAPI endpoint.

A golden evaluation dataset containing 15 questions was created, consisting of 8 simple questions, 4 multi-document or multi-part questions, and 3 unanswerable questions.

RAGAS evaluation showed improvements across all four metrics when comparing answerable questions. Faithfulness improved from 0.90 to 1.00, answer relevancy from 0.7361 to 0.8542, context precision from 0.8379 to 0.9583, and context recall from 0.9167 to 1.00. The faithfulness scores were calculated from 10 valid evaluations in each run because two evaluations returned errors.

The system also achieved 100% abstention accuracy on the three unanswerable questions, demonstrating that it can avoid answering questions when the required information is unavailable.

Overall, the assessment demonstrates a functional RAG pipeline with improved retrieval quality, grounded answers, source citations, and evaluation capabilities. The results indicate that the system is suitable as a foundation for an internal company-policy question-answering application.