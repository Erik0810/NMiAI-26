"""
Improved solver that uses knowledge learned from previous rounds AND
settlement stats (population, food, wealth, defense, factions) from current observations.

Prediction priority:
1. Stats-enhanced model (settlement survival rates, expansion mapping, faction analysis)
2. Learned knowledge base from previous rounds (if available)
3. Empirical terrain distributions from observations
4. Heuristic fallback
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from learn import extract_cell_features, KNOWLEDGE_DIR
from model import build_stats_enhanced_prediction


def load_knowledge():
    """Load cumulative knowledge base from all analyzed rounds."""
    kb_path = os.path.join(KNOWLEDGE_DIR, "cumulative_knowledge.json")
    if not os.path.exists(kb_path):
        print("No knowledge base found. Using default heuristics.")
        return None

    with open(kb_path) as f:
        kb = json.load(f)

    # Merge learnings from all rounds into a single lookup
    merged = {}
    for round_data in kb["rounds"]:
        for key, data in round_data["learnings"].items():
            if key not in merged:
                merged[key] = {"dists": [], "counts": []}
            merged[key]["dists"].append(np.array(data["avg_distribution"]))
            merged[key]["counts"].append(data["count"])

    # Weighted average across rounds (more samples = more weight)
    knowledge = {}
    for key, data in merged.items():
        weights = np.array(data["counts"], dtype=float)
        total = weights.sum()
        if total == 0:
            continue
        avg_dist = sum(d * w for d, w in zip(data["dists"], weights)) / total
        knowledge[key] = avg_dist.tolist()

    print(f"Loaded knowledge: {len(knowledge)} patterns from {len(kb['rounds'])} rounds")
    return knowledge


def build_learned_prediction(class_map, raw_grid, settlements, W, H, knowledge):
    """
    Build prediction using learned distributions instead of hardcoded heuristics.
    Falls back to heuristics for patterns not yet in the knowledge base.
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)

    settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
    port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}

    kb_hits = 0
    kb_misses = 0

    for y in range(H):
        for x in range(W):
            cls = class_map[y, x]

            # Static terrain — always deterministic
            if cls == 5:  # Mountain
                pred[y, x, 5] = 0.95
                continue
            if raw_grid[y, x] == 10:  # Ocean
                pred[y, x, 0] = 0.95
                continue

            # Extract features
            feat = extract_cell_features(
                x, y, class_map, raw_grid, settlement_set, port_set, W, H
            )

            # Build lookup key (same as in learn.py)
            key = _feature_to_key(feat)

            if knowledge and key in knowledge:
                pred[y, x] = np.array(knowledge[key])
                kb_hits += 1
            else:
                # Try progressively less specific keys
                fallback_keys = _generate_fallback_keys(feat)
                found = False
                for fk in fallback_keys:
                    if knowledge and fk in knowledge:
                        pred[y, x] = np.array(knowledge[fk])
                        kb_hits += 1
                        found = True
                        break

                if not found:
                    # Fall back to hardcoded heuristics
                    pred[y, x] = _heuristic_prediction(feat)
                    kb_misses += 1

    # Apply floor and renormalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    print(f"    Knowledge base: {kb_hits} hits, {kb_misses} misses")
    return pred


def _feature_to_key(feat):
    """Convert features to knowledge base lookup key."""
    if feat["is_settlement"] or feat["is_port"]:
        key = (
            "settlement" if not feat["is_port"] else "port",
            "coastal" if feat["is_coastal"] else "inland",
            f"forest_{min(feat['n_forest'], 3)}",
            f"sett_neighbors_{min(feat['n_settlement_neighbors'], 3)}",
        )
    elif feat["class"] == 4:  # Forest
        key = (
            "forest",
            f"near_sett_{min(feat['dist_to_nearest_settlement'], 5)}",
        )
    elif feat["class"] == 0 and feat["raw"] == 11:  # Plains
        key = (
            "plains",
            f"near_sett_{min(feat['dist_to_nearest_settlement'], 5)}",
            "coastal" if feat["is_coastal"] else "inland",
        )
    else:
        key = (f"class_{feat['class']}",)

    return "|".join(str(k) for k in key)


def _generate_fallback_keys(feat):
    """Generate less specific keys to try if exact key not found."""
    keys = []
    if feat["is_settlement"] or feat["is_port"]:
        # Try without neighbor count
        keys.append("|".join([
            "settlement" if not feat["is_port"] else "port",
            "coastal" if feat["is_coastal"] else "inland",
            f"forest_{min(feat['n_forest'], 3)}",
        ]))
        # Try just settlement type + coastal
        keys.append("|".join([
            "settlement" if not feat["is_port"] else "port",
            "coastal" if feat["is_coastal"] else "inland",
        ]))
    elif feat["class"] == 0 and feat["raw"] == 11:
        # Try just plains + distance
        keys.append("|".join([
            "plains",
            f"near_sett_{min(feat['dist_to_nearest_settlement'], 5)}",
        ]))
    return keys


def _heuristic_prediction(feat):
    """Hardcoded heuristic fallback (same as our Round 1 approach)."""
    pred = np.full(NUM_CLASSES, PROB_FLOOR)
    cls = feat["class"]

    if cls == 0:
        pred[0] = 0.85
        if feat["raw"] == 11 and feat["dist_to_nearest_settlement"] <= 3:
            pred[1] = 0.06
            pred[3] = 0.02
            if feat["is_coastal"]:
                pred[2] = 0.03
    elif cls == 4:  # Forest
        pred[4] = 0.80
        if feat["dist_to_nearest_settlement"] <= 2:
            pred[1] = 0.04
    elif cls == 1:  # Settlement
        pred[1] = 0.40
        pred[3] = 0.20
        pred[2] = 0.10 if feat["is_coastal"] else 0.05
        pred[0] = 0.10
    elif cls == 2:  # Port
        pred[2] = 0.45
        pred[1] = 0.15
        pred[3] = 0.20
    elif cls == 3:  # Ruin
        pred[3] = 0.35
        pred[4] = 0.20
        pred[0] = 0.20
        pred[1] = 0.10

    return pred


# ═══════════════════════════════════════════════════════════════════════
# Enhanced query planning with adaptive allocation
# ═══════════════════════════════════════════════════════════════════════

def adaptive_query_plan(client, round_id, detail, total_budget=50):
    """
    Two-phase query strategy:
    Phase 1 (60% budget): Cover all seeds evenly for baseline data
    Phase 2 (40% budget): Focus remaining queries on high-uncertainty areas
    """
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    phase1_budget = int(total_budget * 0.6)  # 30 queries
    phase2_budget = total_budget - phase1_budget  # 20 queries

    queries_per_seed_p1 = phase1_budget // seeds_count  # 6 each

    # Phase 1: Cover each seed evenly
    print("Phase 1: Even coverage")
    phase1_queries = []
    all_observations = []

    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        settlements = state["settlements"]
        vps = _compute_covering_viewports(settlements, W, H)

        seed_queries = []
        for vp in vps[:queries_per_seed_p1]:
            seed_queries.append({"seed_index": seed_idx, **vp})

        # If fewer viewports than budget, repeat
        while len(seed_queries) < queries_per_seed_p1:
            vp = vps[len(seed_queries) % len(vps)]
            seed_queries.append({"seed_index": seed_idx, **vp})

        phase1_queries.extend(seed_queries)

    # Execute phase 1
    for i, q in enumerate(phase1_queries):
        try:
            result = client.simulate(
                round_id=round_id, seed_index=q["seed_index"],
                vx=q["vx"], vy=q["vy"], vw=q["vw"], vh=q["vh"],
            )
            all_observations.append({
                "seed_index": q["seed_index"],
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
            })
            print(f"  P1 [{i+1}/{len(phase1_queries)}] Seed {q['seed_index']} "
                  f"({q['vx']},{q['vy']}) — {len(result['settlements'])} sett")
        except Exception as e:
            print(f"  P1 [{i+1}] FAILED: {e}")
            break

    # Phase 2: Analyze phase 1 results and target high-uncertainty areas
    print(f"\nPhase 2: Targeted queries ({phase2_budget} remaining)")
    phase2_queries = _plan_adaptive_queries(
        all_observations, detail, phase2_budget
    )

    for i, q in enumerate(phase2_queries):
        try:
            result = client.simulate(
                round_id=round_id, seed_index=q["seed_index"],
                vx=q["vx"], vy=q["vy"], vw=q["vw"], vh=q["vh"],
            )
            all_observations.append({
                "seed_index": q["seed_index"],
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
            })
            print(f"  P2 [{i+1}/{len(phase2_queries)}] Seed {q['seed_index']} "
                  f"({q['vx']},{q['vy']}) — {len(result['settlements'])} sett")
        except Exception as e:
            print(f"  P2 [{i+1}] FAILED: {e}")
            break

    return all_observations


def _plan_adaptive_queries(observations, detail, budget):
    """
    Analyze phase 1 observations and plan phase 2 queries targeting:
    - Areas with high variance (settlements that change a lot between runs)
    - Uncovered settlement clusters
    - Seeds with worst coverage
    """
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    # Count observations per cell per seed
    coverage = {}
    variance_score = {}

    for seed_idx in range(seeds_count):
        seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
        cell_classes = defaultdict(list)

        for obs in seed_obs:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            for gy in range(vh):
                for gx in range(vw):
                    mx, my = vx + gx, vy + gy
                    cell_val = obs["grid"][gy][gx]
                    cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                    cell_classes[(seed_idx, mx, my)].append(cls)

        # Find cells with high variance (different outcomes across observations)
        for (si, x, y), classes in cell_classes.items():
            if len(classes) >= 2:
                unique = len(set(classes))
                if unique > 1:
                    variance_score[(si, x, y)] = unique / len(classes)

    # Allocate phase 2 queries
    # Priority: seeds with less coverage, viewports over high-variance areas
    seed_coverage = defaultdict(int)
    for o in observations:
        seed_coverage[o["seed_index"]] += 1

    queries = []
    budget_per_seed = max(1, budget // seeds_count)

    # Sort seeds by least coverage first
    seed_order = sorted(range(seeds_count), key=lambda s: seed_coverage[s])

    remaining = budget
    for seed_idx in seed_order:
        if remaining <= 0:
            break

        state = detail["initial_states"][seed_idx]
        settlements = state["settlements"]
        vps = _compute_covering_viewports(settlements, W, H)

        # Prioritize viewports with high-variance cells
        n = min(budget_per_seed, remaining, len(vps))
        for vp in vps[:n]:
            queries.append({"seed_index": seed_idx, **vp})
            remaining -= 1

    # Fill remaining with repeat queries on densest viewports
    while remaining > 0:
        for seed_idx in seed_order:
            if remaining <= 0:
                break
            state = detail["initial_states"][seed_idx]
            vps = _compute_covering_viewports(state["settlements"], W, H)
            if vps:
                queries.append({"seed_index": seed_idx, **vps[0]})
                remaining -= 1

    return queries


def _compute_covering_viewports(settlements, W, H, vp_size=15):
    """
    Greedy set-cover: find minimal viewports to cover all settlements.
    Optimized: only test viewport positions anchored to settlement coordinates
    instead of brute-force scanning the entire grid.
    """
    alive = [(s["x"], s["y"]) for s in settlements if s["alive"]]
    if not alive:
        return [{"vx": 0, "vy": 0, "vw": vp_size, "vh": vp_size, "n_settlements": 0}]

    uncovered = set(alive)
    viewports = []

    while uncovered:
        best_vp = None
        best_count = 0

        # Generate candidate viewport positions from settlement coords
        candidates = set()
        for sx, sy in uncovered:
            # Try viewport positions that could contain this settlement
            # The settlement can be anywhere within the viewport
            for anchor_x in range(max(0, sx - vp_size + 1), min(W - vp_size + 1, sx + 1)):
                for anchor_y in range(max(0, sy - vp_size + 1), min(H - vp_size + 1, sy + 1)):
                    candidates.add((anchor_x, anchor_y))

        # If too many candidates, subsample using settlement-centered positions
        if len(candidates) > 500:
            candidates = set()
            for sx, sy in uncovered:
                vx = max(0, min(W - vp_size, sx - vp_size // 2))
                vy = max(0, min(H - vp_size, sy - vp_size // 2))
                candidates.add((vx, vy))
                # Also try corners
                for ox in [0, -vp_size // 4, vp_size // 4]:
                    for oy in [0, -vp_size // 4, vp_size // 4]:
                        cx = max(0, min(W - vp_size, sx - vp_size // 2 + ox))
                        cy = max(0, min(H - vp_size, sy - vp_size // 2 + oy))
                        candidates.add((cx, cy))

        for vx, vy in candidates:
            count = sum(1 for (sx, sy) in uncovered
                        if vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)
            if count > best_count:
                best_count = count
                best_vp = {"vx": vx, "vy": vy, "vw": vp_size, "vh": vp_size,
                           "n_settlements": best_count}

        if best_vp is None or best_count == 0:
            break
        viewports.append(best_vp)
        vx, vy = best_vp["vx"], best_vp["vy"]
        uncovered = {(sx, sy) for (sx, sy) in uncovered
                     if not (vx <= sx < vx + vp_size and vy <= sy < vy + vp_size)}

    viewports.sort(key=lambda v: v["n_settlements"], reverse=True)
    return viewports


# ═══════════════════════════════════════════════════════════════════════
# Main improved solver
# ═══════════════════════════════════════════════════════════════════════

def solve():
    """Run the improved solver using learned knowledge."""
    client = AstarClient()

    # Get active round
    active = client.get_active_round()
    if not active:
        print("No active round.")
        return

    round_id = active["id"]
    detail = client.get_round_detail(round_id)
    budget = client.get_budget()
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    remaining = budget["queries_max"] - budget["queries_used"]

    print(f"Round {detail.get('round_number', '?')}: {W}x{H}, {seeds_count} seeds")
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    # Load knowledge base
    knowledge = load_knowledge()

    # Execute adaptive queries
    print(f"\n{'='*60}")
    print("QUERYING")
    print(f"{'='*60}")

    if remaining > 0:
        observations = adaptive_query_plan(client, round_id, detail, remaining)
    else:
        print("No budget! Using learned priors only.")
        observations = []

    # Build predictions using knowledge
    print(f"\n{'='*60}")
    print("BUILDING PREDICTIONS")
    print(f"{'='*60}")

    obs_by_seed = defaultdict(list)
    for obs in observations:
        obs_by_seed[obs["seed_index"]].append(obs)

    predictions = {}
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])

        print(f"\n  Seed {seed_idx}:")

        # Primary: stats-enhanced model (uses settlement stats + factions + expansion)
        pred = build_stats_enhanced_prediction(
            class_map, raw_grid, state["settlements"],
            observations, seed_idx, W, H, knowledge
        )

        # If we have knowledge from previous rounds, blend it in for unobserved cells
        if knowledge:
            pred_kb = build_learned_prediction(
                class_map, raw_grid, state["settlements"], W, H, knowledge
            )
            # For cells NOT observed, blend knowledge base with stats model
            obs_mask = np.zeros((H, W), dtype=bool)
            for obs in obs_by_seed[seed_idx]:
                vp = obs["viewport"]
                vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
                obs_mask[vy:vy+vh, vx:vx+vw] = True

            # For unobserved cells, 50/50 blend between stats model and knowledge base
            for y in range(H):
                for x in range(W):
                    if not obs_mask[y, x]:
                        pred[y, x] = 0.5 * pred[y, x] + 0.5 * pred_kb[y, x]

        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed_idx] = pred

    # Submit
    print(f"\n{'='*60}")
    print("SUBMITTING")
    print(f"{'='*60}")

    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        resp = client.submit(round_id, seed_idx, pred)
        print(f"  Seed {seed_idx}: {resp.get('status', 'ERROR')}")

    print("\nDone! All seeds submitted.")


if __name__ == "__main__":
    solve()
