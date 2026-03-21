"""
Post-round analysis — extract ground truth and learn patterns for future rounds.

After a round completes, this module:
1. Downloads ground truth for all seeds
2. Compares against our predictions (where did we go wrong?)
3. Extracts features per cell and correlates with ground truth outcomes
4. Builds a "knowledge base" of settlement behavior patterns
5. Saves everything for the learned model to use in future rounds

Key insight: hidden parameters are the SAME within a round but DIFFERENT between rounds.
However, the MECHANICS are always the same — so we learn the mapping from
(terrain features) → (outcome distributions) which generalizes across rounds.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES

KNOWLEDGE_DIR = "knowledge_base"


def analyze_round(client: AstarClient, round_id: str, detail: dict):
    """
    Full post-round analysis. Downloads ground truth and extracts learnings.
    Returns analysis dict with all extracted patterns.
    """
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]

    print(f"{'='*60}")
    print(f"POST-ROUND ANALYSIS: Round {detail.get('round_number', '?')}")
    print(f"{'='*60}")

    all_analyses = []
    all_features = []
    all_gt_dists = []

    for seed_idx in range(seeds_count):
        print(f"\n--- Seed {seed_idx} ---")
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            print(f"  Failed to get analysis: {e}")
            continue

        gt = np.array(analysis["ground_truth"])       # H×W×6
        pred = np.array(analysis["prediction"])         # H×W×6
        score = analysis.get("score")
        initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][seed_idx]["grid"])
        class_map = grid_to_class_map(initial_grid.tolist())

        print(f"  Score: {score}")

        # ── Per-cell error analysis ──
        # KL divergence per cell
        kl_per_cell = np.sum(gt * np.log((gt + 1e-10) / (pred + 1e-10)), axis=-1)
        entropy_per_cell = -np.sum(gt * np.log(gt + 1e-10), axis=-1)

        # Dynamic cells (entropy > 0.1)
        dynamic_mask = entropy_per_cell > 0.1
        n_dynamic = np.sum(dynamic_mask)
        avg_kl_dynamic = np.mean(kl_per_cell[dynamic_mask]) if n_dynamic > 0 else 0

        print(f"  Dynamic cells: {n_dynamic}")
        print(f"  Avg KL on dynamic cells: {avg_kl_dynamic:.4f}")

        # ── Worst predictions ──
        flat_idx = np.argsort(kl_per_cell.ravel())[::-1][:10]
        print(f"  Worst 10 cells (highest KL):")
        for idx in flat_idx:
            y, x = divmod(idx, W)
            if kl_per_cell[y, x] < 0.01:
                break
            init_cls = class_map[y, x]
            gt_argmax = np.argmax(gt[y, x])
            pred_argmax = np.argmax(pred[y, x])
            print(f"    ({x},{y}) initial={CLASS_NAMES[init_cls]} "
                  f"gt={CLASS_NAMES[gt_argmax]}({gt[y,x,gt_argmax]:.2f}) "
                  f"pred={CLASS_NAMES[pred_argmax]}({pred[y,x,pred_argmax]:.2f}) "
                  f"KL={kl_per_cell[y,x]:.4f}")

        # ── Extract features for each cell ──
        settlements = detail["initial_states"][seed_idx]["settlements"]
        settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
        port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}

        for y in range(H):
            for x in range(W):
                features = extract_cell_features(
                    x, y, class_map, initial_grid, settlement_set, port_set, W, H
                )
                gt_dist = gt[y, x].tolist()
                all_features.append(features)
                all_gt_dists.append(gt_dist)

        all_analyses.append({
            "seed_index": seed_idx,
            "score": score,
            "n_dynamic": int(n_dynamic),
            "avg_kl_dynamic": float(avg_kl_dynamic),
        })

    # ── Aggregate learnings ──
    learnings = aggregate_learnings(all_features, all_gt_dists)

    # ── Save everything ──
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    knowledge = {
        "round_id": round_id,
        "round_number": detail.get("round_number"),
        "analyses": all_analyses,
        "learnings": learnings,
    }

    # Append to cumulative knowledge
    kb_path = os.path.join(KNOWLEDGE_DIR, "cumulative_knowledge.json")
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            cumulative = json.load(f)
    else:
        cumulative = {"rounds": []}

    cumulative["rounds"].append(knowledge)

    with open(kb_path, "w") as f:
        json.dump(cumulative, f, indent=2)

    # Save detailed feature-gt pairs for model training
    training_path = os.path.join(KNOWLEDGE_DIR, f"training_data_{round_id}.npz")
    np.savez_compressed(
        training_path,
        features=np.array(all_features, dtype=object),
        ground_truth=np.array(all_gt_dists),
    )

    print(f"\n{'='*60}")
    print("LEARNINGS SUMMARY")
    print(f"{'='*60}")
    print_learnings(learnings)
    print(f"\nSaved to {KNOWLEDGE_DIR}/")

    return knowledge


def extract_cell_features(x, y, class_map, raw_grid, settlement_set, port_set, W, H):
    """
    Extract features for a single cell that predict its outcome distribution.
    These features should generalize across rounds (terrain context, not hidden params).
    """
    cls = int(class_map[y, x])
    raw = int(raw_grid[y, x])

    # Is this cell a settlement/port initially?
    is_settlement = (x, y) in settlement_set
    is_port = (x, y) in port_set

    # Neighbor terrain counts (8-connected)
    n_ocean = 0
    n_plains = 0
    n_forest = 0
    n_mountain = 0
    n_settlement = 0
    n_port = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                nc = class_map[ny, nx]
                nr = raw_grid[ny, nx]
                if nr == 10:
                    n_ocean += 1
                elif nr == 11 or nc == 0:
                    n_plains += 1
                if nc == 4:
                    n_forest += 1
                if nc == 5:
                    n_mountain += 1
                if (nx, ny) in settlement_set:
                    n_settlement += 1
                if (nx, ny) in port_set:
                    n_port += 1

    # Distance to nearest settlement
    min_dist = 999
    for (sx, sy), s in settlement_set.items():
        d = abs(x - sx) + abs(y - sy)
        if d < min_dist:
            min_dist = d

    # Number of settlements within radius 5
    settlements_r5 = sum(1 for (sx, sy) in settlement_set
                         if abs(x - sx) + abs(y - sy) <= 5)

    # Edge of map?
    is_edge = x <= 1 or y <= 1 or x >= W - 2 or y >= H - 2

    return {
        "class": cls,
        "raw": raw,
        "is_settlement": is_settlement,
        "is_port": is_port,
        "n_ocean": n_ocean,
        "n_plains": n_plains,
        "n_forest": n_forest,
        "n_mountain": n_mountain,
        "n_settlement_neighbors": n_settlement,
        "n_port_neighbors": n_port,
        "dist_to_nearest_settlement": min_dist,
        "settlements_within_r5": settlements_r5,
        "is_edge": is_edge,
        "is_coastal": n_ocean > 0 and raw != 10,
    }


def aggregate_learnings(all_features, all_gt_dists):
    """
    Aggregate feature-outcome pairs into interpretable patterns.
    Groups cells by type and context, computes average outcome distributions.
    """
    # Group by (initial_class, key_context) → average ground truth distribution
    groups = defaultdict(lambda: {"dists": [], "count": 0})

    for feat, gt in zip(all_features, all_gt_dists):
        cls = feat["class"]
        gt = np.array(gt)

        # Skip trivially static cells
        if cls == 5:  # Mountain
            continue
        if feat["raw"] == 10:  # Ocean
            continue

        # Create context key
        if feat["is_settlement"] or feat["is_port"]:
            # Settlement cells — group by (port_status, coastal, food_access)
            key = (
                "settlement" if not feat["is_port"] else "port",
                "coastal" if feat["is_coastal"] else "inland",
                f"forest_{min(feat['n_forest'], 3)}",
                f"sett_neighbors_{min(feat['n_settlement_neighbors'], 3)}",
            )
        elif cls == 4:  # Forest
            key = (
                "forest",
                f"near_sett_{min(feat['dist_to_nearest_settlement'], 5)}",
            )
        elif cls == 0:  # Empty/Plains
            if feat["raw"] == 11:  # Plains (buildable)
                key = (
                    "plains",
                    f"near_sett_{min(feat['dist_to_nearest_settlement'], 5)}",
                    "coastal" if feat["is_coastal"] else "inland",
                )
            else:
                continue  # Skip generic empty
        else:
            key = (f"class_{cls}",)

        key_str = "|".join(str(k) for k in key)
        groups[key_str]["dists"].append(gt)
        groups[key_str]["count"] += 1

    # Compute average distribution per group
    learnings = {}
    for key_str, group in groups.items():
        dists = np.array(group["dists"])
        avg_dist = np.mean(dists, axis=0).tolist()
        std_dist = np.std(dists, axis=0).tolist()
        learnings[key_str] = {
            "count": group["count"],
            "avg_distribution": avg_dist,
            "std_distribution": std_dist,
        }

    return learnings


def print_learnings(learnings):
    """Pretty-print the key learnings."""
    # Sort by count
    for key, data in sorted(learnings.items(), key=lambda x: -x[1]["count"]):
        if data["count"] < 5:
            continue
        avg = data["avg_distribution"]
        dominant = CLASS_NAMES[np.argmax(avg)]
        entropy = -sum(p * np.log(p + 1e-10) for p in avg)

        if entropy < 0.1:
            continue  # Skip boring static cells

        print(f"  {key} (n={data['count']})")
        print(f"    -> {' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.2f}' for i in range(6))}")
        print(f"    Dominant: {dominant}, Entropy: {entropy:.3f}")


# ═══════════════════════════════════════════════════════════════════════

def run_analysis():
    """Run post-round analysis for the most recently completed round."""
    client = AstarClient()
    my_rounds = client.get_my_rounds()

    # Find completed/scoring rounds
    completed = [r for r in my_rounds if r["status"] in ("completed", "scoring")]
    if not completed:
        print("No completed rounds to analyze yet.")
        print("Available rounds:")
        for r in my_rounds:
            print(f"  Round {r['round_number']}: {r['status']} "
                  f"score={r.get('round_score', 'N/A')}")
        return

    for r in completed:
        print(f"\nRound {r['round_number']}: score={r.get('round_score', 'N/A')} "
              f"rank={r.get('rank', 'N/A')}/{r.get('total_teams', 'N/A')}")

    # Analyze the latest completed round
    latest = completed[-1]
    round_id = latest["id"]

    # Need round detail
    detail = client.get_round_detail(round_id)
    analyze_round(client, round_id, detail)


if __name__ == "__main__":
    run_analysis()
