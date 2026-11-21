
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from src.ingestion.document_loader import load_documents, create_chunks




PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_PATH = PROJECT_ROOT / "data" / "qdrant_storage"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

COLLECTION_NAME = "policy_documents_final"

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def save_chunks(chunks):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_PATH / "chunks.jsonl"

    with open(output_file, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")

    print(f"Saved chunks to: {output_file}")


def create_index(chunks):
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    dimension = embedding_model.get_sentence_embedding_dimension()

    print("Embedding dimensions:", dimension)

    print("\nGenerating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embeddings generated:", len(embeddings))

    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(STORAGE_PATH))

    try:
        # Remove the old collection to avoid stale chunks.
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print("Deleted old Qdrant collection.")

        # Create a fresh collection.
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
        )

        print("Created fresh Qdrant collection.")

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=chunk
                )
            )

        print("\nUploading chunks to Qdrant...")

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        ).count

        print("\nINDEXING COMPLETE")
        print("=================")
        print("Collection:", COLLECTION_NAME)
        print("Vector dimensions:", dimension)
        print("Chunks uploaded:", len(points))
        print("Total stored points:", count)

    finally:
        client.close()


if __name__ == "__main__":
    print("Loading documents...")

    documents = load_documents()

    chunks = create_chunks(documents)

    print("Total chunks:", len(chunks))

    save_chunks(chunks)

    create_index(chunks)
