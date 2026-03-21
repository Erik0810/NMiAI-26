"""
Astar Island Solver — Main pipeline.
Run this to: fetch round data, query strategically, build predictions, submit.
"""

import numpy as np
import json
import os
import time
from collections import defaultdict
from client import AstarClient, grid_to_class_map, build_baseline_prediction
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES

DATA_DIR = "round_data"


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Fetch round data and analyze initial states
# ═══════════════════════════════════════════════════════════════════════

def fetch_round_data(client: AstarClient):
    """Fetch active round details and save locally."""
    active = client.get_active_round()
    if not active:
        print("No active round found. Checking all rounds...")
        rounds = client.get_rounds()
        for r in rounds:
            print(f"  Round {r['round_number']}: {r['status']}")
        return None

    round_id = active["id"]
    print(f"Active round: #{active['round_number']} (ID: {round_id})")
    print(f"  Map: {active['map_width']}x{active['map_height']}")
    print(f"  Closes at: {active.get('closes_at', 'unknown')}")

    detail = client.get_round_detail(round_id)
    budget = client.get_budget()
    print(f"  Seeds: {detail['seeds_count']}")
    print(f"  Budget: {budget['queries_used']}/{budget['queries_max']} used")

    # Save round data
    round_dir = os.path.join(DATA_DIR, round_id)
    save_json(detail, os.path.join(round_dir, "round_detail.json"))
    save_json(budget, os.path.join(round_dir, "budget.json"))

    return detail, budget


def analyze_initial_states(detail: dict):
    """Analyze all seed initial states — find settlements, classify terrain."""
    seeds_count = detail["seeds_count"]
    W, H = detail["map_width"], detail["map_height"]
    print(f"\n{'='*60}")
    print(f"INITIAL STATE ANALYSIS — {W}x{H} map, {seeds_count} seeds")
    print(f"{'='*60}")

    seed_info = []
    for i, state in enumerate(detail["initial_states"]):
        grid = np.array(state["grid"])
        class_map = grid_to_class_map(state["grid"])
        settlements = state["settlements"]

        alive = [s for s in settlements if s["alive"]]
        ports = [s for s in alive if s["has_port"]]
        inland = [s for s in alive if not s["has_port"]]

        # Find settlement cluster centers (for viewport targeting)
        if alive:
            sx = [s["x"] for s in alive]
            sy = [s["y"] for s in alive]
            cx, cy = np.mean(sx), np.mean(sy)
        else:
            cx, cy = W // 2, H // 2

        # Count terrain types
        unique, counts = np.unique(class_map, return_counts=True)
        terrain_dist = dict(zip(unique.tolist(), counts.tolist()))

        info = {
            "seed_index": i,
            "settlements": settlements,
            "alive_count": len(alive),
            "port_count": len(ports),
            "inland_count": len(inland),
            "center": (cx, cy),
            "terrain_dist": terrain_dist,
            "class_map": class_map,
            "raw_grid": grid,
        }
        seed_info.append(info)

        print(f"\nSeed {i}:")
        print(f"  Settlements: {len(alive)} alive ({len(ports)} ports, {len(inland)} inland)")
        print(f"  Terrain: {terrain_dist}")
        print(f"  Settlement cluster center: ({cx:.1f}, {cy:.1f})")
        for s in alive:
            tag = "PORT" if s["has_port"] else "TOWN"
            print(f"    [{tag}] ({s['x']}, {s['y']})")

    return seed_info


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Strategic viewport planning
# ═══════════════════════════════════════════════════════════════════════

def find_dynamic_regions(seed_info: dict, W: int, H: int):
    """
    Identify cells that are likely dynamic (settlements + their neighborhoods).
    Returns a set of (x, y) tuples marking cells worth observing.
    """
    dynamic = set()
    settlements = seed_info["settlements"]
    class_map = seed_info["class_map"]

    for s in settlements:
        if not s["alive"]:
            continue
        sx, sy = s["x"], s["y"]
        # Settlement itself + expansion radius (settlements can expand ~3-5 cells)
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    # Only count non-ocean, non-mountain cells
                    if class_map[ny, nx] != 5:  # not mountain
                        dynamic.add((nx, ny))
    return dynamic


def plan_viewports(seed_info: list, W: int, H: int, total_budget: int = 50):
    """
    Plan optimal viewport placements across all seeds.
    Strategy:
    - Pick 2 focus seeds (most settlements) — 10 queries each for repeat sampling
    - 3 remaining seeds — 6 queries each for coverage + calibration
    - Reserve ~8 queries for adaptive follow-up
    """
    # Sort seeds by settlement count (most interesting first)
    by_interest = sorted(seed_info, key=lambda s: s["alive_count"], reverse=True)

    # Focus seeds get more queries for empirical distribution building
    focus_seeds = [by_interest[0]["seed_index"], by_interest[1]["seed_index"]]
    other_seeds = [s["seed_index"] for s in by_interest[2:]]

    print(f"\n{'='*60}")
    print("QUERY STRATEGY")
    print(f"{'='*60}")
    print(f"Focus seeds (repeat sampling): {focus_seeds}")
    print(f"Other seeds (coverage): {other_seeds}")

    viewports = []

    for si in seed_info:
        idx = si["seed_index"]
        settlements = [s for s in si["settlements"] if s["alive"]]
        if not settlements:
            continue

        # Compute optimal viewport placements to cover all settlements
        seed_viewports = compute_covering_viewports(settlements, W, H)

        if idx in focus_seeds:
            # Repeat top viewports for empirical distribution
            n_queries = 10
            repeated = []
            while len(repeated) < n_queries:
                for vp in seed_viewports:
                    if len(repeated) >= n_queries:
                        break
                    repeated.append(vp)
            seed_vps = repeated
        else:
            # Just cover each area once
            seed_vps = seed_viewports[:6]

        for vp in seed_vps:
            viewports.append({"seed_index": idx, **vp})

        print(f"  Seed {idx}: {len(seed_vps)} queries planned "
              f"({len(seed_viewports)} unique viewports)")

    print(f"  Total planned: {len(viewports)} queries")
    return viewports


def compute_covering_viewports(settlements: list, W: int, H: int, vp_size: int = 15):
    """
    Compute minimal set of 15x15 viewports that cover all settlements.
    Uses greedy set-cover approach.
    """
    uncovered = set()
    for s in settlements:
        uncovered.add((s["x"], s["y"]))

    viewports = []
    while uncovered:
        # Find viewport placement that covers the most uncovered settlements
        best_vp = None
        best_count = 0

        # Try centering viewport on each uncovered settlement
        for sx, sy in list(uncovered):
            # Center viewport on this settlement
            vx = max(0, min(W - vp_size, sx - vp_size // 2))
            vy = max(0, min(H - vp_size, sy - vp_size // 2))
            count = sum(1 for (px, py) in uncovered
                        if vx <= px < vx + vp_size and vy <= py < vy + vp_size)
            if count > best_count:
                best_count = count
                best_vp = {"vx": vx, "vy": vy, "vw": vp_size, "vh": vp_size}

        if best_vp is None:
            break

        viewports.append(best_vp)
        # Remove covered settlements
        vx, vy = best_vp["vx"], best_vp["vy"]
        uncovered = {(px, py) for (px, py) in uncovered
                     if not (vx <= px < vx + vp_size and vy <= py < vy + vp_size)}

    return viewports


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Execute queries and collect observations
# ═══════════════════════════════════════════════════════════════════════

def execute_queries(client: AstarClient, round_id: str, viewports: list,
                    max_queries: int = 50):
    """Execute planned viewport queries and store results."""
    budget = client.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\n{'='*60}")
    print(f"EXECUTING QUERIES — {remaining} remaining")
    print(f"{'='*60}")

    # Limit to available budget
    to_execute = viewports[:remaining]

    results = []  # list of (seed_index, viewport, result)
    for i, vp in enumerate(to_execute):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=vp["seed_index"],
                vx=vp["vx"], vy=vp["vy"],
                vw=vp["vw"], vh=vp["vh"],
            )
            results.append({
                "seed_index": vp["seed_index"],
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
                "queries_used": result["queries_used"],
            })
            print(f"  [{i+1}/{len(to_execute)}] Seed {vp['seed_index']} "
                  f"@ ({vp['vx']},{vp['vy']}) — "
                  f"{len(result['settlements'])} settlements, "
                  f"budget: {result['queries_used']}/{result['queries_max']}")
        except Exception as e:
            print(f"  [{i+1}/{len(to_execute)}] FAILED: {e}")
            break

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Build predictions from observations
# ═══════════════════════════════════════════════════════════════════════

def build_predictions(seed_info: list, observations: list,
                      W: int, H: int, seeds_count: int) -> dict:
    """
    Build H×W×6 prediction tensors for all seeds.
    Combines: baseline from initial state + empirical distributions from queries.
    """
    print(f"\n{'='*60}")
    print("BUILDING PREDICTIONS")
    print(f"{'='*60}")

    # Group observations by seed
    obs_by_seed = defaultdict(list)
    for obs in observations:
        obs_by_seed[obs["seed_index"]].append(obs)

    # Also collect cross-seed observations for transfer learning
    all_settlement_outcomes = []  # (initial_class, final_class, context) tuples

    predictions = {}
    for seed_idx in range(seeds_count):
        info = seed_info[seed_idx]
        class_map = info["class_map"]

        # Start with baseline prediction
        pred = build_baseline_prediction(class_map)

        # Count observations per cell
        obs_counts = np.zeros((H, W, NUM_CLASSES))
        obs_total = np.zeros((H, W))

        for obs in obs_by_seed[seed_idx]:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            grid = np.array(obs["grid"])

            for gy in range(vh):
                for gx in range(vw):
                    map_x = vx + gx
                    map_y = vy + gy
                    if 0 <= map_x < W and 0 <= map_y < H:
                        cell_val = grid[gy][gx]
                        cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                        obs_counts[map_y, map_x, cls] += 1
                        obs_total[map_y, map_x] += 1

        # For cells with observations, blend empirical distribution with baseline
        observed_mask = obs_total > 0
        n_observed = np.sum(observed_mask)
        print(f"  Seed {seed_idx}: {len(obs_by_seed[seed_idx])} observations, "
              f"{n_observed} cells observed")

        for y in range(H):
            for x in range(W):
                if obs_total[y, x] > 0:
                    n = obs_total[y, x]
                    empirical = obs_counts[y, x] / n

                    # Blend: more observations = more trust in empirical
                    # With 1 observation, alpha = 0.6; with 5+, alpha = 0.9
                    alpha = min(0.95, 0.5 + 0.1 * n)
                    pred[y, x] = alpha * empirical + (1 - alpha) * pred[y, x]

        # Apply floor and renormalize
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)

        predictions[seed_idx] = pred

    # Cross-seed transfer: for seeds with few observations, borrow patterns
    _cross_seed_transfer(predictions, seed_info, obs_by_seed, W, H)

    return predictions


def _cross_seed_transfer(predictions: dict, seed_info: list,
                         obs_by_seed: dict, W: int, H: int):
    """
    Transfer learned patterns from well-observed seeds to less-observed ones.
    Key insight: hidden parameters are the SAME across all seeds.
    
    For each settlement in a less-observed seed, find a similar settlement
    (same port status, similar neighbor terrain) in a well-observed seed
    and borrow its outcome distribution.
    """
    # Collect outcome distributions from observed settlements
    settlement_profiles = []  # (has_port, n_forest_neighbors, n_ocean_neighbors, distribution)

    for seed_idx, obs_list in obs_by_seed.items():
        if len(obs_list) < 3:
            continue  # Need enough observations for reliable distributions

        info = seed_info[seed_idx]
        class_map = info["class_map"]

        for s in info["settlements"]:
            if not s["alive"]:
                continue
            sx, sy = s["x"], s["y"]

            # Count neighbor terrain
            n_forest = 0
            n_ocean = 0
            n_mountain = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        c = class_map[ny, nx]
                        if c == 4:
                            n_forest += 1
                        elif c == 0:
                            # Check if ocean
                            raw = info["raw_grid"][ny, nx]
                            if raw == 10:
                                n_ocean += 1
                        elif c == 5:
                            n_mountain += 1

            # Get observed distribution at this settlement
            dist = predictions[seed_idx][sy, sx].copy()

            settlement_profiles.append({
                "has_port": s["has_port"],
                "n_forest": n_forest,
                "n_ocean": n_ocean,
                "n_mountain": n_mountain,
                "dist": dist,
            })

    if not settlement_profiles:
        return

    print(f"  Cross-seed transfer: {len(settlement_profiles)} settlement profiles collected")

    # Apply profiles to less-observed seeds
    for seed_idx in range(len(seed_info)):
        if len(obs_by_seed[seed_idx]) >= 3:
            continue  # Already well-observed

        info = seed_info[seed_idx]
        class_map = info["class_map"]
        pred = predictions[seed_idx]

        for s in info["settlements"]:
            if not s["alive"]:
                continue
            sx, sy = s["x"], s["y"]

            n_forest = sum(1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                           if (dx != 0 or dy != 0)
                           and 0 <= sx + dx < W and 0 <= sy + dy < H
                           and class_map[sy + dy, sx + dx] == 4)
            n_ocean = sum(1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                          if (dx != 0 or dy != 0)
                          and 0 <= sx + dx < W and 0 <= sy + dy < H
                          and info["raw_grid"][sy + dy, sx + dx] == 10)

            # Find most similar profile
            best_match = None
            best_score = float("inf")
            for p in settlement_profiles:
                score = (abs(p["n_forest"] - n_forest) +
                         abs(p["n_ocean"] - n_ocean) +
                         (0 if p["has_port"] == s["has_port"] else 3))
                if score < best_score:
                    best_score = score
                    best_match = p

            if best_match is not None:
                # Blend: 40% from similar settlement, 60% from baseline
                pred[sy, sx] = 0.4 * best_match["dist"] + 0.6 * pred[sy, sx]

        # Renormalize
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed_idx] = pred


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Enhance predictions with neighborhood-aware heuristics  
# ═══════════════════════════════════════════════════════════════════════

def enhance_predictions_with_neighborhood(predictions: dict, seed_info: list,
                                           W: int, H: int):
    """
    Improve predictions for unobserved cells using neighborhood heuristics:
    - Empty cells near settlements might become settlements (expansion)
    - Forest near ruins might reclaim them
    - Coastal settlements more likely to become ports
    """
    print(f"\n  Applying neighborhood heuristics...")

    for seed_idx, pred in predictions.items():
        info = seed_info[seed_idx]
        class_map = info["class_map"]
        raw_grid = info["raw_grid"]

        # Identify cells adjacent to settlements
        settlement_positions = set()
        for s in info["settlements"]:
            if s["alive"]:
                settlement_positions.add((s["x"], s["y"]))

        for y in range(H):
            for x in range(W):
                cls = class_map[y, x]

                # Check if adjacent to any settlement
                near_settlement = False
                settlement_dist = float("inf")
                for sx, sy in settlement_positions:
                    d = abs(x - sx) + abs(y - sy)
                    if d < settlement_dist:
                        settlement_dist = d
                    if d <= 2:
                        near_settlement = True

                # Check if coastal (adjacent to ocean)
                is_coastal = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny_ = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny_ < H:
                            if raw_grid[ny_, nx] == 10:
                                is_coastal = True

                # Plains near settlements — might get settled
                if cls == 0 and raw_grid[y, x] == 11:
                    if near_settlement:
                        pred[y, x, 1] += 0.08  # settlement expansion
                        if is_coastal:
                            pred[y, x, 2] += 0.04  # could become port
                        pred[y, x, 3] += 0.03  # might become ruin after expansion
                    elif settlement_dist <= 5:
                        pred[y, x, 1] += 0.03
                        pred[y, x, 3] += 0.01

                # Forest near ruin — reclamation possible
                if cls == 4 and near_settlement:
                    # Forest might be cleared / settlement might expand here
                    pred[y, x, 1] += 0.02

                # Settlement on coast — more likely to become port
                if cls == 1 and is_coastal:
                    pred[y, x, 2] += 0.08

        # Renormalize
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed_idx] = pred


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run():
    client = AstarClient()

    # Phase 1: Fetch and analyze
    result = fetch_round_data(client)
    if result is None:
        print("No active round. Exiting.")
        return
    detail, budget = result
    round_id = detail["id"]
    W, H = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]

    seed_info = analyze_initial_states(detail)

    # Phase 2: Plan queries
    viewports = plan_viewports(seed_info, W, H, budget["queries_max"])

    # Phase 3: Execute queries
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining > 0:
        observations = execute_queries(client, round_id, viewports, remaining)
    else:
        print("No budget remaining! Using baseline predictions only.")
        observations = []

    # Save observations
    round_dir = os.path.join(DATA_DIR, round_id)
    save_json(observations, os.path.join(round_dir, "observations.json"))

    # Phase 4: Build predictions
    predictions = build_predictions(seed_info, observations, W, H, seeds_count)

    # Phase 5: Enhance with neighborhood heuristics
    enhance_predictions_with_neighborhood(predictions, seed_info, W, H)

    # Phase 6: Submit
    print(f"\n{'='*60}")
    print("SUBMITTING PREDICTIONS")
    print(f"{'='*60}")
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        resp = client.submit(round_id, seed_idx, pred)
        print(f"  Seed {seed_idx}: {resp.get('status', 'unknown')}")

    print(f"\nDone! All {seeds_count} seeds submitted.")
    print("Check scores at app.ainm.no after the round closes.")


if __name__ == "__main__":
    run()
