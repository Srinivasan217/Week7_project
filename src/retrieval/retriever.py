from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT /".env")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "policy_documents_final"
STORAGE_PATH = PROJECT_ROOT / "data"/"qdrant_storage"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=str(STORAGE_PATH))

def retrieve_chunks(question, k=4):
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True

    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        metadata = payload.get("metadata", {})

        if not metadata:
            metadata = {
                key: value
                for key, value in payload.items()
                if key != "text"
            }

        results.append({
            "id": str(point.id),
            "text": payload.get("text", ""),
            "score": point.score,
            "metadata": metadata
        })

    return results
