"""
API client for Astar Island challenge.
Wraps all REST endpoints with proper auth and error handling.
"""

import requests
import time
import json
import numpy as np
from config import BASE_URL, ACCESS_TOKEN, TERRAIN_TO_CLASS, PROB_FLOOR


class AstarClient:
    def __init__(self, token: str = None):
        self.session = requests.Session()
        token = token or ACCESS_TOKEN
        if not token:
            raise ValueError("No access token provided. Set ACCESS_TOKEN in config.py")
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.base = f"{BASE_URL}/astar-island"
        self._last_request_time = 0

    def _rate_limit(self, min_interval: float = 1.1):
        """Enforce rate limiting (conservative ~1 req/s to avoid 429)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    # ─── Round Management ────────────────────────────────────────────

    def get_rounds(self) -> list:
        """List all rounds."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/rounds")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_active_round(self) -> dict | None:
        """Get the currently active round, or None."""
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> dict:
        """Get full round details including initial states."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/rounds/{round_id}")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_budget(self) -> dict:
        """Check remaining query budget."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/budget")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    # ─── Simulation ──────────────────────────────────────────────────

    def simulate(self, round_id: str, seed_index: int,
                 vx: int = 0, vy: int = 0, vw: int = 15, vh: int = 15) -> dict:
        """
        Run one stochastic simulation and observe through viewport.
        Costs 1 query from budget.
        """
        for attempt in range(5):
            self._rate_limit()
            resp = self.session.post(f"{self.base}/simulate", json={
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": vx,
                "viewport_y": vy,
                "viewport_w": vw,
                "viewport_h": vh,
            })
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()  # raise the last 429
        return resp.json()

    # ─── Prediction Submission ───────────────────────────────────────

    def submit(self, round_id: str, seed_index: int, prediction: np.ndarray) -> dict:
        """
        Submit H×W×6 prediction tensor for one seed.
        Applies probability floor and renormalizes automatically.
        """
        # Safety: enforce floor and renormalize
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        for attempt in range(5):
            self._rate_limit(min_interval=1.5)
            resp = self.session.post(f"{self.base}/submit", json={
                "round_id": round_id,
                "seed_index": seed_index,
                "prediction": prediction.tolist(),
            })
            if resp.status_code == 429:
                wait = 2 ** attempt + 2
                print(f"    Submit rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    # ─── Analysis ────────────────────────────────────────────────────

    def get_my_rounds(self) -> list:
        """Get all rounds with team scores and budget info."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/my-rounds")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_my_predictions(self, round_id: str) -> list:
        """Get submitted predictions with argmax/confidence grids."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/my-predictions/{round_id}")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        """Post-round ground truth comparison (only after round completes)."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/analysis/{round_id}/{seed_index}")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_leaderboard(self) -> list:
        """Public leaderboard."""
        for attempt in range(5):
            self._rate_limit(0.5)
            resp = self.session.get(f"{self.base}/leaderboard")
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()


def grid_to_class_map(grid: list[list[int]]) -> np.ndarray:
    """Convert raw terrain grid (internal codes) to class indices (0-5)."""
    arr = np.array(grid)
    class_map = np.zeros_like(arr)
    for code, cls in TERRAIN_TO_CLASS.items():
        class_map[arr == code] = cls
    return class_map


def build_baseline_prediction(class_map: np.ndarray) -> np.ndarray:
    """
    Build a baseline prediction from the initial state class map.
    Static terrain gets high confidence; settlements get moderate confidence
    since they can change over 50 years.
    """
    H, W = class_map.shape
    pred = np.full((H, W, 6), PROB_FLOOR)

    for y in range(H):
        for x in range(W):
            cls = class_map[y, x]
            if cls == 0:  # Empty (ocean/plains)
                # Ocean/plains mostly stay empty, small chance of settlement expansion
                pred[y, x, 0] = 0.90
            elif cls == 5:  # Mountain — never changes
                pred[y, x, 5] = 0.95
            elif cls == 4:  # Forest — mostly stable, can be displaced near settlements
                pred[y, x, 4] = 0.85
            elif cls == 1:  # Settlement — can survive, become port, or collapse to ruin
                pred[y, x, 1] = 0.45
                pred[y, x, 3] = 0.20  # ruin
                pred[y, x, 2] = 0.10  # port
                pred[y, x, 0] = 0.10  # empty
            elif cls == 2:  # Port — can survive or collapse
                pred[y, x, 2] = 0.45
                pred[y, x, 1] = 0.15  # downgrade to settlement
                pred[y, x, 3] = 0.20  # ruin
                pred[y, x, 0] = 0.05
            elif cls == 3:  # Ruin — can be reclaimed or overgrown
                pred[y, x, 3] = 0.35
                pred[y, x, 4] = 0.20  # forest reclaims
                pred[y, x, 0] = 0.20  # fades to plains
                pred[y, x, 1] = 0.10  # reclaimed as settlement

    # Renormalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred
