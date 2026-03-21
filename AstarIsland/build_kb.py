"""
Build survival-indexed knowledge base from all completed rounds' GT data.

Key insight: each round has different hidden parameters that dramatically affect
settlement survival rates (from 0% in Round 3 to 90% in Round 1). A fixed KB
that averages across rounds is suboptimal — we need distributions indexed by
the survival regime, so we can interpolate based on what we observe.

Output: knowledge_base/survival_indexed_kb.json
{
    "round_survival_rates": {round_num: avg_survival_rate, ...},
    "context_keys": {
        key: {
            "rates": [sorted survival rates],
            "dists": [corresponding H×6 distribution arrays]
        }
    }
}
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from learn import extract_cell_features, KNOWLEDGE_DIR


def build_survival_indexed_kb():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    print(f"Building survival-indexed KB from {len(completed)} rounds")
    
    # Step 1: Compute per-round, per-seed survival rates from GT
    round_data = {}  # round_num -> {survival_rate, features_and_dists}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        all_survival_probs = []  # GT settlement+port prob for each initial settlement
        seed_features = []  # (features, gt_dist) for all cells across all seeds
        
        for seed_idx in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, seed_idx)
            except Exception as e:
                print(f"  Round {rn} seed {seed_idx}: FAILED ({e})")
                continue
            
            gt = np.array(analysis["ground_truth"])
            initial_grid = np.array(
                analysis.get("initial_grid") or detail["initial_states"][seed_idx]["grid"]
            )
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][seed_idx]["settlements"]
            settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
            port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}
            
            # Compute GT survival probability for each initial settlement
            for (sx, sy), sdata in settlement_set.items():
                surv_prob = float(gt[sy, sx, 1] + gt[sy, sx, 2])  # settlement + port
                all_survival_probs.append(surv_prob)
            
            # Extract features for all dynamic cells
            for y in range(H):
                for x in range(W):
                    cls = int(class_map[y, x])
                    raw = int(initial_grid[y, x])
                    if cls == 5 or raw == 10:
                        continue  # skip static
                    
                    features = extract_cell_features(
                        x, y, class_map, initial_grid, settlement_set, port_set, W, H
                    )
                    gt_dist = gt[y, x].tolist()
                    seed_features.append((features, gt_dist))
        
        if all_survival_probs:
            avg_survival = float(np.mean(all_survival_probs))
        else:
            avg_survival = 0.0
        
        round_data[rn] = {
            "survival_rate": avg_survival,
            "features": seed_features,
        }
        print(f"  Round {rn}: survival_rate={avg_survival:.4f} "
              f"({len(all_survival_probs)} settlements, {len(seed_features)} cells)")
    
    # Step 2: Group by context key, indexed by round survival rate
    context_groups = defaultdict(lambda: defaultdict(list))
    # context_groups[key][round_num] = list of gt_dists
    
    for rn, data in round_data.items():
        for features, gt_dist in data["features"]:
            key = _make_context_key(features)
            if key is None:
                continue
            context_groups[key][rn].append(np.array(gt_dist))
    
    # Step 3: Build the survival-indexed KB
    survival_rates = {rn: data["survival_rate"] for rn, data in round_data.items()}
    
    kb = {
        "round_survival_rates": survival_rates,
        "context_keys": {}
    }
    
    for key, round_dists in context_groups.items():
        rates = []
        dists = []
        counts = []
        
        for rn in sorted(round_dists.keys()):
            rate = survival_rates[rn]
            avg_dist = np.mean(round_dists[rn], axis=0).tolist()
            n = len(round_dists[rn])
            rates.append(rate)
            dists.append(avg_dist)
            counts.append(n)
        
        # Sort by survival rate
        sorted_idx = np.argsort(rates)
        rates = [rates[i] for i in sorted_idx]
        dists = [dists[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]
        
        kb["context_keys"][key] = {
            "rates": rates,
            "dists": dists,
            "counts": counts,
        }
    
    # Save
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    kb_path = os.path.join(KNOWLEDGE_DIR, "survival_indexed_kb.json")
    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2)
    
    print(f"\nSaved survival-indexed KB: {len(kb['context_keys'])} context keys")
    print(f"Round survival rates: {survival_rates}")
    
    # Print sample entries
    print(f"\n--- Sample entries ---")
    for key in ["settlement|inland|forest_3", "settlement|inland|forest_0",
                 "plains|near_sett_1|inland", "plains|near_sett_5|inland",
                 "forest|near_sett_1", "forest|near_sett_5",
                 "port|coastal|forest_1",
                 "plains|near_sett_2|coastal"]:
        if key in kb["context_keys"]:
            entry = kb["context_keys"][key]
            print(f"\n  {key}:")
            for rate, dist, cnt in zip(entry["rates"], entry["dists"], entry["counts"]):
                print(f"    surv={rate:.3f} (n={cnt:4d}): "
                      f"{' '.join(f'{CLASS_NAMES[i]}:{dist[i]:.3f}' for i in range(6))}")
    
    # Test interpolation
    print(f"\n--- Interpolation test ---")
    test_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for key in ["settlement|inland|forest_3", "plains|near_sett_1|inland", "forest|near_sett_1"]:
        if key not in kb["context_keys"]:
            continue
        entry = kb["context_keys"][key]
        print(f"\n  {key}:")
        for test_rate in test_rates:
            dist = interpolate_distribution(entry["rates"], entry["dists"], test_rate)
            print(f"    surv={test_rate:.1f}: "
                  f"{' '.join(f'{CLASS_NAMES[i]}:{dist[i]:.3f}' for i in range(6))}")
    
    return kb


def _make_context_key(features):
    """Create context key from cell features — with extended distance bins."""
    cls = features["class"]
    raw = features["raw"]
    
    if features["is_port"]:
        coast = "coastal" if features["is_coastal"] else "inland"
        return f"port|{coast}|forest_{min(features['n_forest'], 3)}"
    elif features["is_settlement"]:
        coast = "coastal" if features["is_coastal"] else "inland"
        return f"settlement|{coast}|forest_{min(features['n_forest'], 3)}"
    elif cls == 4:  # Forest
        dist = min(features["dist_to_nearest_settlement"], 10)
        if features["is_coastal"]:
            return f"forest|near_sett_{dist}|coastal"
        return f"forest|near_sett_{dist}"
    elif raw == 11:  # Plains
        coast = "coastal" if features["is_coastal"] else "inland"
        dist = min(features["dist_to_nearest_settlement"], 10)
        return f"plains|near_sett_{dist}|{coast}"
    
    return None


def interpolate_distribution(rates, dists, target_rate):
    """
    Linearly interpolate distribution based on target survival rate.
    rates: sorted list of survival rates
    dists: corresponding distributions (list of 6-element lists)
    target_rate: the estimated survival rate for current round
    """
    rates = np.array(rates)
    dists = np.array(dists)
    
    if len(rates) == 1:
        return dists[0].copy()
    
    # Clamp to range
    if target_rate <= rates[0]:
        return dists[0].copy()
    if target_rate >= rates[-1]:
        return dists[-1].copy()
    
    # Find bracketing indices
    idx = np.searchsorted(rates, target_rate) - 1
    idx = max(0, min(idx, len(rates) - 2))
    
    # Linear interpolation
    r0, r1 = rates[idx], rates[idx + 1]
    if r1 - r0 < 1e-10:
        return dists[idx].copy()
    
    t = (target_rate - r0) / (r1 - r0)
    result = (1 - t) * dists[idx] + t * dists[idx + 1]
    
    # Ensure valid distribution
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    
    return result


if __name__ == "__main__":
    build_survival_indexed_kb()
