
from src.generation.answerer import hybrid_retrieve

question = (
    "What do the policy documents say about communicating "
    "guidelines and reviewing policies?"
)

results = hybrid_retrieve(question, k=4)

print("\nRETRIEVAL TEST")
print("=" * 60)

for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"File: {result['metadata']['filename']}")
    print(f"Text: {result['text']}")
