"""
Continue R16 — load existing observation, run remaining queries, resubmit.
"""
import numpy as np
import json
import os
import time
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model_v4 import (
    build_prediction_v4, load_survival_kb, load_expansion_kb,
    estimate_survival_mle, estimate_expansion_from_obs,
)
from solve_v4 import greedy_cover


def main():
    client = AstarClient()
    
    budget = client.get_budget()
    round_id = budget["round_id"]
    used = budget["queries_used"]
    total = budget["queries_max"]
    remaining = total - used
    print(f"Budget: {used}/{total} used, {remaining} remaining")
    
    detail = client.get_round_detail(round_id)
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    rn = detail.get("round_number", "?")
    print(f"Round {rn}: {W}x{H}, {seeds_count} seeds, closes: {detail.get('closes_at', '?')}")
    
    # Load existing observations
    obs_path = f"round_data/observations_{round_id}.json"
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            observations = json.load(f)
        print(f"Loaded {len(observations)} existing observations")
    else:
        observations = []
    
    # Load KBs
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    # Run remaining queries
    if remaining > 0:
        print(f"\n=== Running {remaining} remaining queries ===")
        
        # Plan queries across all seeds
        queries_per_seed = remaining // seeds_count
        leftover = remaining - queries_per_seed * seeds_count
        
        # Track which seeds already have observations
        existing_seeds = set(o["seed_index"] for o in observations)
        
        all_queries = []
        for seed_idx in range(seeds_count):
            state = detail["initial_states"][seed_idx]
            settlements = state["settlements"]
            alive = [(s["x"], s["y"]) for s in settlements if s["alive"]]
            
            viewports = greedy_cover(alive, W, H)
            n_vps = len(viewports)
            
            seed_budget = queries_per_seed + (1 if seed_idx < leftover else 0)
            
            # Skip viewports we might already have (approximate)
            existing_for_seed = sum(1 for o in observations if o["seed_index"] == seed_idx)
            start_vp = existing_for_seed
            
            seed_queries = []
            for i in range(seed_budget):
                vp = viewports[(start_vp + i) % n_vps]
                seed_queries.append({"seed_index": seed_idx, **vp})
            
            all_queries.extend(seed_queries)
        
        print(f"Planned {len(all_queries)} queries across {seeds_count} seeds")
        
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
                    print(f"  [{i+1}/{len(all_queries)}] seed={q['seed_index']} "
                          f"budget={result['queries_used']}/{result['queries_max']}")
            except Exception as e:
                print(f"  [{i+1}] FAILED: {e}")
                if "budget" in str(e).lower():
                    break
                # Rate limit — wait and retry
                if "429" in str(e):
                    print("    Waiting 10s for rate limit...")
                    time.sleep(10)
        
        # Save updated observations
        os.makedirs("round_data", exist_ok=True)
        with open(obs_path, "w") as f:
            json.dump(observations, f, indent=2)
        print(f"\nSaved {len(observations)} total observations")
    
    # Count per-seed observations
    per_seed = {}
    for o in observations:
        si = o["seed_index"]
        per_seed[si] = per_seed.get(si, 0) + 1
    print(f"Observations per seed: {per_seed}")
    
    # Estimate hidden parameters
    print(f"\n=== Parameter Estimation ===")
    all_seeds_settlements = {}
    all_seeds_initial_states = {}
    for si in range(seeds_count):
        all_seeds_settlements[si] = detail["initial_states"][si]["settlements"]
        all_seeds_initial_states[si] = detail["initial_states"][si]
    
    est_survival, confident = estimate_survival_mle(
        observations, all_seeds_initial_states, W, H, survival_kb
    )
    est_expansion, exp_confident = estimate_expansion_from_obs(
        observations, all_seeds_initial_states, W, H
    )
    print(f"  Survival: {est_survival:.4f} (confident={confident})")
    print(f"  Expansion: {est_expansion:.4f} (confident={exp_confident})")
    
    # Build predictions
    print(f"\n=== Building Predictions ===")
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
        predictions[seed_idx] = pred
        
        argmax = np.argmax(pred, axis=-1)
        for c in range(NUM_CLASSES):
            n = np.sum(argmax == c)
            print(f"    {CLASS_NAMES[c]:12s}: {n:4d} cells")
    
    # Submit
    print(f"\n=== Submitting ===")
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        resp = client.submit(round_id, seed_idx, pred)
        status = resp.get("status", "ERROR")
        score = resp.get("score", "N/A")
        print(f"  Seed {seed_idx}: {status} (score: {score})")
    
    print(f"\nDONE — R{rn} submitted with V4 + full observations!")


if __name__ == "__main__":
    main()
