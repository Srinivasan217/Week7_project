
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")

url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

payload = {
    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "messages": [
        {
            "role": "user",
            "content": "What is 2 + 2? Answer briefly."
        }
    ],
    "max_tokens": 256,
    "temperature": 0.2,
    "stream": False
}

print("Testing NVIDIA API...", flush=True)

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=90
    )

    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    print("Model:", result.get("model"))
    print("Answer:", answer)

except requests.exceptions.RequestException as error:
    print("NVIDIA API request failed:", error)
    if error.response is not None:
        print("Response details:", error.response.text)
