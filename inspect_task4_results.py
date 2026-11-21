
import json
from pathlib import Path

file_path = Path(
    "data/processed/task4_ragas_dataset_hybrid_final.json"
)

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("DATASET TYPE:", type(data).__name__)

if isinstance(data, list):
    print("TOTAL RECORDS:", len(data))

    for index, item in enumerate(data):
        if index in [2, 3, 6, 10, 13]:
            print("\n" + "=" * 70)
            print("QUESTION RECORD:", index + 1)
            print("=" * 70)

            for key, value in item.items():
                print(f"\n{key}:")
                print(value)
else:
    print(data)
