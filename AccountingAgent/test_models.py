"""Test which Gemini models are available."""
import requests
import json

# Read OAuth token
with open("token.txt") as f:
    token = f.read().strip()

headers = {"Authorization": f"Bearer {token}"}

# 1. Try listing models via Generative Language API
print("=== Generative Language API (Google AI Studio) ===")
url = "https://generativelanguage.googleapis.com/v1beta/models"
try:
    r = requests.get(url, headers=headers, timeout=20)
    print(f"List status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        models = data.get("models", [])
        for m in models:
            name = m.get("name", "")
            if "3." in name or "pro" in name:
                print(f"  {name}")
    else:
        print(f"  {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Try generating with specific model names on Generative Language API
print("\n=== Direct model test (Generative Language API) ===")
body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "Say hi"}]}]})
test_models = [
    "gemini-3.1-pro-preview-high",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro",
    "gemini-3.1-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
for m in test_models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
    try:
        r = requests.post(url, headers=headers, data=body, timeout=20)
        if r.status_code == 200:
            print(f"  {m:45s} => OK!")
        else:
            msg = ""
            try:
                msg = r.json().get("error", {}).get("message", "")[:80]
            except:
                pass
            print(f"  {m:45s} => {r.status_code} {msg}")
    except Exception as e:
        print(f"  {m:45s} => Error: {e}")

# 3. Try Vertex AI with fresh patterns
print("\n=== Vertex AI (europe-west1) ===")
body2 = json.dumps({"contents": [{"role": "user", "parts": [{"text": "Say hi"}]}]})
for m in test_models:
    url = f"https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/ainm26osl-764/locations/europe-west1/publishers/google/models/{m}:generateContent"
    try:
        r = requests.post(url, headers={**headers, "Content-Type": "application/json"}, 
                         data=body2, timeout=20)
        if r.status_code in [200, 400]:
            print(f"  {m:45s} => EXISTS ({r.status_code})")
        else:
            print(f"  {m:45s} => {r.status_code}")
    except Exception as e:
        print(f"  {m:45s} => Error: {e}")
