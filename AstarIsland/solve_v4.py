"""
V4 Solver — Adaptive Bayesian + 2D KB solver.

Architecture:
1. Run 50 queries distributed across 5 seeds (10 each)
2. Estimate survival + expansion rates from ALL observations
3. Use V4 model: 2D KB interpolation + adaptive Bayesian blending
4. Submit all 5 seeds
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model_v4 import (
    build_prediction_v4, load_survival_kb, load_expansion_kb,
    estimate_survival_mle, estimate_expansion_from_obs,
    DEFAULT_SURVIVAL_RATE, DEFAULT_EXPANSION_RATE,
)


def solve():
    """Main solver using V4 model."""
    client = AstarClient()
    
    active = client.get_active_round()
    if not active:
        print("No active round found!")
        for r in client.get_rounds():
            print(f"  Round {r['round_number']}: {r['status']}")
        return
    
    round_id = active["id"]
    detail = client.get_round_detail(round_id)
    budget = client.get_budget()
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    total = budget["queries_max"]
    used = budget["queries_used"]
    remaining = total - used
    
    print(f"{'='*60}")
    print(f"ROUND {detail.get('round_number', '?')} — V4 Solver")
    print(f"{'='*60}")
    print(f"  Map: {W}x{H}, Seeds: {seeds_count}")
    print(f"  Budget: {used}/{total} used, {remaining} remaining")
    print(f"  Closes: {detail.get('closes_at', 'unknown')}")
    
    # Load knowledge bases
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    # ── Plan and execute queries ──
    print(f"\n{'='*60}")
    print("OBSERVATIONS")
    print(f"{'='*60}")
    
    obs_path = f"round_data/observations_{round_id}.json"
    if remaining <= 0 and os.path.exists(obs_path):
        print(f"  Budget exhausted — loading saved observations")
        with open(obs_path) as f:
            observations = json.load(f)
        print(f"  Loaded {len(observations)} observations")
    elif remaining > 0:
        observations = execute_queries(client, round_id, detail, remaining)
        os.makedirs("round_data", exist_ok=True)
        with open(obs_path, "w") as f:
            json.dump(observations, f, indent=2)
        print(f"\nSaved {len(observations)} observations")
    else:
        observations = []
        print(f"  No budget and no saved observations!")
    
    # ── Estimate hidden parameters ──
    print(f"\n{'='*60}")
    print("PARAMETER ESTIMATION")
    print(f"{'='*60}")
    
    all_seeds_settlements = {}
    all_seeds_initial_states = {}
    for si in range(seeds_count):
        all_seeds_settlements[si] = detail["initial_states"][si]["settlements"]
        all_seeds_initial_states[si] = detail["initial_states"][si]
    
    # Survival rate (MLE, much more precise)
    est_survival, confident = estimate_survival_mle(
        observations, all_seeds_initial_states, W, H, survival_kb
    )
    
    # Expansion rate (from observations)
    est_expansion, exp_confident = estimate_expansion_from_obs(
        observations, all_seeds_initial_states, W, H
    )
    
    print(f"  Survival: {est_survival:.4f} (confident={confident})")
    print(f"  Expansion: {est_expansion:.4f} (confident={exp_confident})")
    
    # ── Build predictions ──
    print(f"\n{'='*60}")
    print("BUILDING PREDICTIONS")
    print(f"{'='*60}")
    
    predictions = {}
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])
        
        print(f"\n  Seed {seed_idx}:")
        pred = build_prediction_v4(
            class_map, raw_grid, state["settlements"],
            observations, seed_idx, W, H,
            survival_kb=survival_kb,
            expansion_kb=expansion_kb,
            estimated_survival=est_survival,
            estimated_expansion=est_expansion,
            all_seeds_settlements=all_seeds_settlements,
        )
        
        assert pred.shape == (H, W, NUM_CLASSES)
        assert np.all(pred >= PROB_FLOOR - 0.001)
        assert np.allclose(pred.sum(axis=-1), 1.0, atol=0.01)
        
        predictions[seed_idx] = pred
        
        argmax = np.argmax(pred, axis=-1)
        for c in range(NUM_CLASSES):
            n = np.sum(argmax == c)
            print(f"    {CLASS_NAMES[c]:12s}: {n:4d} cells")
    
    # ── Submit ──
    print(f"\n{'='*60}")
    print("SUBMITTING")
    print(f"{'='*60}")
    
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        resp = client.submit(round_id, seed_idx, pred)
        status = resp.get("status", "ERROR")
        score = resp.get("score", "N/A")
        print(f"  Seed {seed_idx}: {status} (score: {score})")
    
    print(f"\n{'='*60}")
    print("DONE — All seeds submitted with V4!")
    print(f"{'='*60}")


def execute_queries(client, round_id, detail, budget):
    """Execute queries with settlement cluster coverage."""
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    
    if budget <= 0:
        return []
    
    queries_per_seed = budget // seeds_count
    leftover = budget - queries_per_seed * seeds_count
    
    all_queries = []
    
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        settlements = state["settlements"]
        alive = [(s["x"], s["y"]) for s in settlements if s["alive"]]
        
        seed_budget = queries_per_seed + (1 if seed_idx < leftover else 0)
        
        # Strategy: full grid coverage first, then settlement focus
        grid_vps = full_grid_viewports(W, H)
        settlement_vps = greedy_cover(alive, W, H)
        
        seed_queries = []
        
        # Phase 1: Full grid coverage (9 viewports for 40x40 map)
        for vp in grid_vps[:seed_budget]:
            seed_queries.append({"seed_index": seed_idx, **vp})
        
        # Phase 2: Settlement-focused extra viewports
        remaining = seed_budget - len(seed_queries)
        if remaining > 0:
            for vp in settlement_vps[:remaining]:
                seed_queries.append({"seed_index": seed_idx, **vp})
        
        all_queries.extend(seed_queries)
    
    observations = []
    for i, q in enumerate(all_queries):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=q["seed_index"],
                vx=q["vx"], vy=q["vy"],
                vw=q["vw"], vh=q["vh"],
            )
            observations.append({
                "seed_index": q["seed_index"],
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
            })
            if (i + 1) % 10 == 0 or i == len(all_queries) - 1:
                print(f"  [{i+1}/{len(all_queries)}] Budget: {result['queries_used']}/{result['queries_max']}")
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}")
            if "429" in str(e) or "budget" in str(e).lower():
                break
    
    return observations


def full_grid_viewports(W, H, vp_size=15):
    """
    Systematic grid tiling to cover the entire WxH map.
    For 40x40 with 15x15 viewports: 3x3 = 9 viewports covering everything.
    Positions are chosen to maximize coverage with minimal gaps.
    """
    viewports = []
    
    # Calculate positions: spread viewports to cover full grid
    # Number of viewports needed per axis
    n_x = max(1, -(-W // vp_size))  # ceiling division
    n_y = max(1, -(-H // vp_size))
    
    # Calculate step sizes (with overlap when grid doesn't divide evenly)
    if n_x == 1:
        x_positions = [0]
    else:
        step_x = (W - vp_size) / (n_x - 1) if n_x > 1 else 0
        x_positions = [max(0, min(W - vp_size, int(round(i * step_x)))) for i in range(n_x)]
    
    if n_y == 1:
        y_positions = [0]
    else:
        step_y = (H - vp_size) / (n_y - 1) if n_y > 1 else 0
        y_positions = [max(0, min(H - vp_size, int(round(i * step_y)))) for i in range(n_y)]
    
    for vy in y_positions:
        for vx in x_positions:
            viewports.append({"vx": vx, "vy": vy, "vw": vp_size, "vh": vp_size})
    
    return viewports


def greedy_cover(positions, W, H, vp_size=15):
    """Greedy set-cover for viewport placement."""
    if not positions:
        return [{"vx": 0, "vy": 0, "vw": vp_size, "vh": vp_size}]
    
    uncovered = set(positions)
    viewports = []
    
    while uncovered:
        best_vp = None
        best_count = 0
        
        candidates = set()
        for sx, sy in uncovered:
            vx = max(0, min(W - vp_size, sx - vp_size // 2))
            vy = max(0, min(H - vp_size, sy - vp_size // 2))
            candidates.add((vx, vy))
            for ox in [-vp_size//3, 0, vp_size//3]:
                for oy in [-vp_size//3, 0, vp_size//3]:
                    cx = max(0, min(W - vp_size, sx - vp_size//2 + ox))
                    cy = max(0, min(H - vp_size, sy - vp_size//2 + oy))
                    candidates.add((cx, cy))
        
        for vx, vy in candidates:
            count = sum(1 for (sx, sy) in uncovered
                       if vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)
            if count > best_count:
                best_count = count
                best_vp = {"vx": vx, "vy": vy, "vw": vp_size, "vh": vp_size,
                          "n_settlements": count}
        
        if best_vp is None or best_count == 0:
            break
        
        viewports.append(best_vp)
        vx, vy = best_vp["vx"], best_vp["vy"]
        uncovered = {(sx, sy) for (sx, sy) in uncovered
                    if not (vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)}
    
    viewports.sort(key=lambda v: v["n_settlements"], reverse=True)
    return viewports


if __name__ == "__main__":
    solve()
