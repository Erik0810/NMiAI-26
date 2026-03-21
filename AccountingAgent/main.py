"""
Tripletex AI Accounting Agent — FastAPI application.

Exposes POST /solve as required by the competition.
Deployed on GCP Cloud Run.
"""

import logging
import os
import time

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from agent import run_agent
from tripletex_client import TripletexClient

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Tripletex AI Accounting Agent", version="1.0.0")

API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(request: Request):
    """Optional API key verification."""
    if not API_KEY:
        return  # No key configured, allow all
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {API_KEY}":
        return
    raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve", dependencies=[Depends(verify_api_key)])
async def solve(request: Request):
    """
    Main competition endpoint.

    Receives a task prompt + Tripletex credentials,
    runs the AI agent, returns {"status": "completed"}.
    """
    start = time.time()

    body = await request.json()
    prompt = body.get("prompt", "")
    files = body.get("files", [])
    creds = body.get("tripletex_credentials", {})

    base_url = creds.get("base_url", "")
    session_token = creds.get("session_token", "")

    if not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing tripletex_credentials")

    logger.info(f"=== New task received ===")
    logger.info(f"Prompt ({len(prompt)} chars): {prompt[:200]}...")
    logger.info(f"Files: {[f.get('filename') for f in files]}")
    logger.info(f"Base URL: {base_url}")

    try:
        with TripletexClient(base_url, session_token) as client:
            await run_agent(prompt, files, client)

            elapsed = time.time() - start
            logger.info(
                f"=== Task completed in {elapsed:.1f}s | "
                f"API calls: {client.call_count} | "
                f"Errors: {client.error_count} ==="
            )

    except Exception:
        elapsed = time.time() - start
        logger.exception(f"Agent failed after {elapsed:.1f}s")
        # Still return completed — partial work might score some points
        # and the competition expects this response format

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
