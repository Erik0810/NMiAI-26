"""
Round 1 solver — optimized for this specific round.
40x40 map, 5 seeds, 50 queries, settlements ranging from 30-60 per seed.

Strategy:
- 10 queries per seed (50 / 5 = 10 each)
- For each seed: plan ~3-4 unique viewports covering settlement clusters
- Repeat the densest viewports 2-3x to build empirical distributions
- Build predictions from initial state + observations
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map, build_baseline_prediction
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES

# ═══════════════════════════════════════════════════════════════════════

def load_round():
    with open("round_detail.json") as f:
        return json.load(f)


def plan_seed_viewports(settlements, W=40, H=40, vp_size=15):
    """
    Plan viewports to cover all settlements for one seed.
    Returns list of {vx, vy, vw, vh, n_settlements} sorted by density.
    """
    alive = [(s["x"], s["y"]) for s in settlements if s["alive"]]
    if not alive:
        return []

    # Try a grid of possible viewport positions and score by settlement coverage
    best_viewports = []
    uncovered = set(alive)

    while uncovered:
        best_vp = None
        best_count = 0
        best_pos = None

        # Try all possible positions (step by 1 for precision)
        for vy in range(0, H - vp_size + 1, 2):
            for vx in range(0, W - vp_size + 1, 2):
                count = sum(1 for (sx, sy) in uncovered
                            if vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)
                if count > best_count:
                    best_count = count
                    best_pos = (vx, vy)

        if best_pos is None or best_count == 0:
            break

        vx, vy = best_pos
        vp = {"vx": vx, "vy": vy, "vw": vp_size, "vh": vp_size, "n_settlements": best_count}
        best_viewports.append(vp)
        uncovered = {(sx, sy) for (sx, sy) in uncovered
                     if not (vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)}

    # Sort by density (most settlements first)
    best_viewports.sort(key=lambda v: v["n_settlements"], reverse=True)
    return best_viewports


def allocate_queries(seed_viewports_map, total_budget=50, seeds_count=5):
    """
    Allocate queries across seeds. Each seed gets ~10 queries.
    Within a seed, prioritize viewports by settlement density.
    Repeat dense viewports for better empirical distributions.
    """
    queries_per_seed = total_budget // seeds_count  # 10 each

    all_queries = []
    for seed_idx in range(seeds_count):
        viewports = seed_viewports_map[seed_idx]
        if not viewports:
            continue

        seed_queries = []
        budget = queries_per_seed

        # First pass: cover each viewport at least once
        for vp in viewports:
            if budget <= 0:
                break
            seed_queries.append({"seed_index": seed_idx, **vp})
            budget -= 1

        # Second pass: repeat densest viewports
        vp_idx = 0
        while budget > 0 and viewports:
            vp = viewports[vp_idx % len(viewports)]
            seed_queries.append({"seed_index": seed_idx, **vp})
            budget -= 1
            vp_idx += 1

        all_queries.extend(seed_queries)

    return all_queries


def execute_all_queries(client, round_id, queries):
    """Execute all planned queries."""
    results = []
    for i, q in enumerate(queries):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=q["seed_index"],
                vx=q["vx"], vy=q["vy"],
                vw=q["vw"], vh=q["vh"],
            )
            results.append({
                "seed_index": q["seed_index"],
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
            })
            n_sett = len(result["settlements"])
            used = result["queries_used"]
            mx = result["queries_max"]
            print(f"  [{i+1}/{len(queries)}] Seed {q['seed_index']} "
                  f"({q['vx']},{q['vy']}) {q['vw']}x{q['vh']} — "
                  f"{n_sett} settlements — budget {used}/{mx}")
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] FAILED: {e}")
            # If budget exhausted, stop
            if "429" in str(e) or "budget" in str(e).lower():
                break

    return results


def build_all_predictions(detail, observations):
    """Build predictions for all seeds combining baseline + observations."""
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    # Group observations by seed
    obs_by_seed = defaultdict(list)
    for obs in observations:
        obs_by_seed[obs["seed_index"]].append(obs)

    predictions = {}

    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])

        # Start with baseline
        pred = build_baseline_prediction(class_map)

        # Build observation counts
        obs_counts = np.zeros((H, W, NUM_CLASSES))
        obs_total = np.zeros((H, W))

        for obs in obs_by_seed[seed_idx]:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            grid = obs["grid"]

            for gy in range(vh):
                for gx in range(vw):
                    mx = vx + gx
                    my = vy + gy
                    if 0 <= mx < W and 0 <= my < H:
                        cell_val = grid[gy][gx]
                        cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                        obs_counts[my, mx, cls] += 1
                        obs_total[my, mx] += 1

        # Blend empirical with baseline
        n_observed = int(np.sum(obs_total > 0))
        n_multi = int(np.sum(obs_total > 1))
        print(f"  Seed {seed_idx}: {len(obs_by_seed[seed_idx])} queries, "
              f"{n_observed} cells observed, {n_multi} cells with 2+ observations")

        for y in range(H):
            for x in range(W):
                if obs_total[y, x] > 0:
                    n = obs_total[y, x]
                    empirical = obs_counts[y, x] / n

                    # More observations → more trust in empirical
                    # 1 obs: alpha=0.6, 2: 0.7, 3: 0.8, 5+: 0.9+
                    alpha = min(0.95, 0.5 + 0.1 * n)
                    pred[y, x] = alpha * empirical + (1 - alpha) * pred[y, x]

        # ── Neighborhood heuristics for unobserved dynamic cells ──
        settlement_positions = set()
        for s in state["settlements"]:
            if s["alive"]:
                settlement_positions.add((s["x"], s["y"]))

        for y in range(H):
            for x in range(W):
                if obs_total[y, x] > 0:
                    continue  # Already handled by empirical data

                cls = class_map[y, x]

                # Skip static terrain
                if cls == 5:  # Mountain
                    continue
                if raw_grid[y, x] == 10:  # Ocean
                    continue

                # Distance to nearest settlement
                min_dist = float("inf")
                for sx, sy in settlement_positions:
                    d = abs(x - sx) + abs(y - sy)
                    if d < min_dist:
                        min_dist = d

                # Check coastal
                is_coastal = any(
                    0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y + dy, x + dx] == 10
                    for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                    if dx != 0 or dy != 0
                )

                # Plains/empty near settlements → expansion possible
                if cls == 0 and raw_grid[y, x] == 11:  # Plains
                    if min_dist <= 2:
                        pred[y, x, 1] += 0.12  # settlement
                        if is_coastal:
                            pred[y, x, 2] += 0.06  # port
                        pred[y, x, 3] += 0.04  # ruin (expand then collapse)
                    elif min_dist <= 4:
                        pred[y, x, 1] += 0.05
                        pred[y, x, 3] += 0.02

                # Settlement on coast → might become port
                if cls == 1 and is_coastal:
                    pred[y, x, 2] += 0.10

        # ── Apply floor and renormalize ──
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed_idx] = pred

    # ── Cross-seed transfer for less-observed seeds ──
    cross_seed_enhance(predictions, detail, obs_by_seed)

    return predictions


def cross_seed_enhance(predictions, detail, obs_by_seed):
    """Transfer patterns from well-observed seeds to less-observed ones."""
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    # Collect settlement outcome profiles from all observed seeds
    profiles = []
    for seed_idx in range(seeds_count):
        if len(obs_by_seed[seed_idx]) < 3:
            continue
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])

        for s in state["settlements"]:
            if not s["alive"]:
                continue
            sx, sy = s["x"], s["y"]

            # Neighbor features
            n_forest = 0
            n_ocean = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if class_map[ny, nx] == 4:
                            n_forest += 1
                        if raw_grid[ny, nx] == 10:
                            n_ocean += 1

            profiles.append({
                "has_port": s["has_port"],
                "n_forest": n_forest,
                "n_ocean": n_ocean,
                "dist": predictions[seed_idx][sy, sx].copy(),
            })

    if not profiles:
        return

    print(f"  Cross-seed: {len(profiles)} profiles for transfer")

    # Apply to all seeds' unobserved settlements
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])
        pred = predictions[seed_idx]

        for s in state["settlements"]:
            if not s["alive"]:
                continue
            sx, sy = s["x"], s["y"]

            n_forest = sum(
                1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if (dx != 0 or dy != 0) and 0 <= sx + dx < W and 0 <= sy + dy < H
                and grid_to_class_map(state["grid"])[sy + dy, sx + dx] == 4
            )
            n_ocean = sum(
                1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if (dx != 0 or dy != 0) and 0 <= sx + dx < W and 0 <= sy + dy < H
                and raw_grid[sy + dy, sx + dx] == 10
            )

            # Find best matching profile
            best = None
            best_score = float("inf")
            for p in profiles:
                score = (abs(p["n_forest"] - n_forest) +
                         abs(p["n_ocean"] - n_ocean) +
                         (0 if p["has_port"] == s["has_port"] else 3))
                if score < best_score:
                    best_score = score
                    best = p

            if best:
                # Blend 30% transfer, 70% existing
                pred[sy, sx] = 0.3 * best["dist"] + 0.7 * pred[sy, sx]

        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed_idx] = pred


def submit_all(client, round_id, predictions, seeds_count):
    """Submit all predictions."""
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        # Final safety: floor + renorm
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        resp = client.submit(round_id, seed_idx, pred)
        print(f"  Seed {seed_idx}: {resp.get('status', 'ERROR')}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    detail = load_round()
    round_id = detail["id"]
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    client = AstarClient()
    budget = client.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Round {detail['round_number']}: {W}x{H}, {seeds_count} seeds")
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    # ── Phase 1: Plan viewports for each seed ──
    print(f"\n{'='*60}")
    print("PHASE 1: Planning viewports")
    print(f"{'='*60}")

    seed_viewports = {}
    for i, state in enumerate(detail["initial_states"]):
        vps = plan_seed_viewports(state["settlements"], W, H)
        seed_viewports[i] = vps
        covers = sum(v["n_settlements"] for v in vps)
        alive = sum(1 for s in state["settlements"] if s["alive"])
        print(f"  Seed {i}: {len(vps)} viewports needed to cover {alive} settlements")
        for j, vp in enumerate(vps):
            print(f"    VP{j}: ({vp['vx']},{vp['vy']}) — {vp['n_settlements']} settlements")

    # ── Phase 2: Allocate and execute queries ──
    print(f"\n{'='*60}")
    print("PHASE 2: Executing queries")
    print(f"{'='*60}")

    queries = allocate_queries(seed_viewports, remaining, seeds_count)
    print(f"  Planned {len(queries)} queries")

    observations = execute_all_queries(client, round_id, queries)

    # Save observations
    os.makedirs("round_data", exist_ok=True)
    with open(f"round_data/observations_{round_id}.json", "w") as f:
        json.dump(observations, f, indent=2)

    # ── Phase 3: Build predictions ──
    print(f"\n{'='*60}")
    print("PHASE 3: Building predictions")
    print(f"{'='*60}")

    predictions = build_all_predictions(detail, observations)

    # ── Phase 4: Submit ──
    print(f"\n{'='*60}")
    print("PHASE 4: Submitting predictions")
    print(f"{'='*60}")

    submit_all(client, round_id, predictions, seeds_count)

    # Summary
    print(f"\n{'='*60}")
    print("DONE — All seeds submitted!")
    print(f"{'='*60}")
    print(f"Round closes at: {detail.get('closes_at', 'unknown')}")
    print("Check scores at: https://app.ainm.no")


if __name__ == "__main__":
    main()
