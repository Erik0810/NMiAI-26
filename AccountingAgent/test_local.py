"""
Local test script — sends a sample task to your local /solve endpoint.

Usage:
  1. Set SANDBOX_BASE_URL and SANDBOX_SESSION_TOKEN in .env or here
  2. Start the server: python main.py
  3. Run: python test_local.py
"""

import json
import os
import requests

# ── Config ───────────────────────────────────────────────────────────
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8080/solve")

SANDBOX_BASE_URL = os.environ.get("SANDBOX_BASE_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
SANDBOX_SESSION_TOKEN = os.environ.get("SANDBOX_SESSION_TOKEN", "")

# ── Sample tasks to test with ────────────────────────────────────────
SAMPLE_TASKS = [
    {
        "name": "Create employee (Norwegian)",
        "prompt": "Opprett en ansatt med fornavn 'Ola' og etternavn 'Nordmann'. E-postadressen er ola@example.com. Han skal være kontoadministrator.",
    },
    {
        "name": "Create customer (English)",
        "prompt": "Create a customer named 'Acme AS' with email 'post@acme.no'.",
    },
    {
        "name": "Create product (Norwegian)",
        "prompt": "Opprett et produkt med navn 'Konsulenttime' og pris 1500 kr eksklusiv mva.",
    },
]


def run_test(task: dict):
    print(f"\n{'='*60}")
    print(f"Test: {task['name']}")
    print(f"Prompt: {task['prompt']}")
    print(f"{'='*60}")

    payload = {
        "prompt": task["prompt"],
        "files": [],
        "tripletex_credentials": {
            "base_url": SANDBOX_BASE_URL,
            "session_token": SANDBOX_SESSION_TOKEN,
        },
    }

    try:
        resp = requests.post(AGENT_URL, json=payload, timeout=300)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to agent. Is it running? (python main.py)")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    if not SANDBOX_SESSION_TOKEN:
        print("WARNING: SANDBOX_SESSION_TOKEN is empty.")
        print("Set it via environment variable or edit this file.")
        print("Get your token from: https://app.ainm.no")
        print()

    # Run all sample tasks
    for task in SAMPLE_TASKS:
        run_test(task)
        print()
