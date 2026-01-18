"""
client_demo.py

Goal:
- Send a sample request to the running FastAPI service
- Print the response

Run:
1) Start server: uvicorn src.app.main:app --reload --port 8000
2) In another terminal: python -m src.app.client_demo
"""

from __future__ import annotations

import json
import urllib.request


def main() -> None:
    url = "http://127.0.0.1:8000/predict"
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print("Response:", body)


if __name__ == "__main__":
    main()
