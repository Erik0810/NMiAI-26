"""
V3 Solver — Survival-rate-indexed interpolation model.

Architecture:
1. Run 50 queries distributed across 5 seeds (10 each)
2. Estimate round-wide survival rate from ALL observations (hidden params are shared)
3. Interpolate per-cell distributions from survival-indexed KB
4. Adjust individual settlement predictions based on per-cell observations
5. Submit all 5 seeds

Key Design Decisions:
- Survival rate is the master parameter — determines which distributions to use
- Observations serve TWO purposes: (a) estimate survival rate, (b) per-settlement adjustment
- KB from Rounds 1-5 provides survival-indexed distributions for each terrain context
- All seeds share the same hidden params → aggregate observations across seeds
"""

import numpy as np
import json
import os
import glob
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model_v3 import build_prediction_v3, load_survival_kb, _estimate_survival_rate, estimate_survival_mle


def solve():
    """Main solver using V3 survival-indexed model."""
    client = AstarClient()
    
    # Get active round
    active = client.get_active_round()
    if not active:
        print("No active round found!")
        print("Available rounds:")
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
    print(f"ROUND {detail.get('round_number', '?')}")
    print(f"{'='*60}")
    print(f"  Map: {W}x{H}, Seeds: {seeds_count}")
    print(f"  Budget: {used}/{total} used, {remaining} remaining")
    print(f"  Closes: {detail.get('closes_at', 'unknown')}")
    
    # Save round detail
    with open("round_detail.json", "w") as f:
        json.dump(detail, f, indent=2)
    
    # Load survival-indexed knowledge base
    survival_kb = load_survival_kb()
    
    # ── Plan and execute queries ──
    print(f"\n{'='*60}")
    print("EXECUTING QUERIES")
    print(f"{'='*60}")
    
    # Try loading existing observations for this round (if queries were already used)
    obs_path = f"round_data/observations_{round_id}.json"
    if remaining <= 0 and os.path.exists(obs_path):
        print(f"  Budget exhausted — loading saved observations from {obs_path}")
        with open(obs_path) as f:
            observations = json.load(f)
        print(f"  Loaded {len(observations)} observations")
    elif remaining > 0:
        observations = execute_queries(client, round_id, detail, remaining)
        
        # Save observations
        os.makedirs("round_data", exist_ok=True)
        with open(obs_path, "w") as f:
            json.dump(observations, f, indent=2)
        print(f"\nSaved {len(observations)} observations to {obs_path}")
    else:
        observations = []
        print(f"  No budget and no saved observations!")
    
    # ── Estimate survival rate from ALL seeds' observations ──
    print(f"\n{'='*60}")
    print("SURVIVAL ESTIMATION")
    print(f"{'='*60}")
    
    all_seeds_settlements = {}
    all_seeds_initial_states = {}
    for si in range(seeds_count):
        all_seeds_settlements[si] = detail["initial_states"][si]["settlements"]
        all_seeds_initial_states[si] = detail["initial_states"][si]
    
    # Use MLE estimation (much more precise than simple counting)
    est_survival, confident = estimate_survival_mle(
        observations, all_seeds_initial_states, W, H, survival_kb
    )
    print(f"  Final estimated survival rate: {est_survival:.4f} (confident={confident})")
    
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
        pred = build_prediction_v3(
            class_map, raw_grid, state["settlements"],
            observations, seed_idx, W, H,
            survival_kb=survival_kb,
            estimated_survival=est_survival,
            all_seeds_settlements=all_seeds_settlements,
        )
        
        # Sanity check
        assert pred.shape == (H, W, NUM_CLASSES)
        assert np.all(pred >= PROB_FLOOR - 0.001)
        assert np.allclose(pred.sum(axis=-1), 1.0, atol=0.01)
        
        predictions[seed_idx] = pred
        
        # Print summary
        argmax = np.argmax(pred, axis=-1)
        for c in range(NUM_CLASSES):
            n = np.sum(argmax == c)
            print(f"    {CLASS_NAMES[c]:12s}: {n:4d} cells")
    
    # ── Submit predictions ──
    print(f"\n{'='*60}")
    print("SUBMITTING")
    print(f"{'='*60}")
    
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        resp = client.submit(round_id, seed_idx, pred)
        status = resp.get("status", "ERROR")
        score = resp.get("score", "N/A")
        print(f"  Seed {seed_idx}: {status} (estimated score: {score})")
    
    print(f"\n{'='*60}")
    print("DONE — All seeds submitted!")
    print(f"{'='*60}")
    print(f"  Closes: {detail.get('closes_at', 'unknown')}")
    print(f"  Check: https://app.ainm.no")


def execute_queries(client, round_id, detail, budget):
    """
    Execute queries with focus on settlement cluster coverage and repetition.
    
    Strategy:
    - Identify settlement clusters per seed
    - Cover each cluster with a viewport
    - Repeat the densest viewports to get 3+ observations
    - Allocate budget evenly across seeds
    """
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    
    if budget <= 0:
        print("  No budget remaining!")
        return []
    
    queries_per_seed = budget // seeds_count
    leftover = budget - queries_per_seed * seeds_count
    
    all_queries = []
    
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        settlements = state["settlements"]
        alive = [(s["x"], s["y"]) for s in settlements if s["alive"]]
        
        # Compute covering viewports
        viewports = greedy_cover(alive, W, H)
        n_vps = len(viewports)
        
        # Allocate queries: cover + repeat
        seed_budget = queries_per_seed + (1 if seed_idx < leftover else 0)
        
        seed_queries = []
        # Phase 1: cover each viewport once
        for vp in viewports[:seed_budget]:
            seed_queries.append({"seed_index": seed_idx, **vp})
        
        # Phase 2: repeat from densest viewports
        if len(seed_queries) < seed_budget:
            vp_idx = 0
            while len(seed_queries) < seed_budget:
                vp = viewports[vp_idx % n_vps]
                seed_queries.append({"seed_index": seed_idx, **vp})
                vp_idx += 1
        
        all_queries.extend(seed_queries)
        
        seen_count = len(set((q["vx"], q["vy"]) for q in seed_queries))
        repeat_count = len(seed_queries) - seen_count
        print(f"  Seed {seed_idx}: {len(alive)} settlements, {n_vps} viewports, "
              f"{seen_count} unique, {repeat_count} repeats ({seed_budget} total)")
    
    # Execute all queries
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
                b = result["queries_used"]
                bm = result["queries_max"]
                print(f"  [{i+1}/{len(all_queries)}] Budget: {b}/{bm}")
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}")
            if "429" in str(e) or "budget" in str(e).lower():
                break
    
    return observations


def greedy_cover(positions, W, H, vp_size=15):
    """
    Greedy set-cover: find minimal viewports to cover all positions.
    Uses settlement-anchored candidates for efficiency.
    """
    if not positions:
        return [{"vx": 0, "vy": 0, "vw": vp_size, "vh": vp_size}]
    
    uncovered = set(positions)
    viewports = []
    
    while uncovered:
        best_vp = None
        best_count = 0
        
        # Generate candidate viewport positions centered on uncovered positions
        candidates = set()
        for sx, sy in uncovered:
            # Try viewport positions that contain this position
            vx = max(0, min(W - vp_size, sx - vp_size // 2))
            vy = max(0, min(H - vp_size, sy - vp_size // 2))
            candidates.add((vx, vy))
            # Also try offset positions
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
