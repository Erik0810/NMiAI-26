"""
Build expansion-indexed knowledge base (2D: survival + expansion).

For each context key, stores distributions indexed by BOTH:
- survival_rate (average settlement survival probability)
- expansion_rate (fraction of non-settlement plains that become settlement/port)

This captures the second hidden parameter that survival rate alone misses
(e.g., R7 vs R2: same survival but different expansion → very different oracle scores).
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from learn import extract_cell_features, KNOWLEDGE_DIR
from build_kb import interpolate_distribution


def build_expansion_kb():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    print(f"Building expansion-indexed KB from {len(completed)} rounds")
    
    round_data = {}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        all_survival_probs = []
        all_expansion_probs = []  # Per non-settlement plains cell: P(settlement|plains)
        seed_features = []
        
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
            
            for (sx, sy), sdata in settlement_set.items():
                surv_prob = float(gt[sy, sx, 1] + gt[sy, sx, 2])
                all_survival_probs.append(surv_prob)
            
            for y in range(H):
                for x in range(W):
                    cls = int(class_map[y, x])
                    raw = int(initial_grid[y, x])
                    if cls == 5 or raw == 10:
                        continue
                    
                    features = extract_cell_features(
                        x, y, class_map, initial_grid, settlement_set, port_set, W, H
                    )
                    gt_dist = gt[y, x].tolist()
                    seed_features.append((features, gt_dist))
                    
                    # Expansion: non-settlement plains becoming settlement
                    if raw == 11 and (x, y) not in settlement_set:
                        exp_prob = float(gt[y, x, 1] + gt[y, x, 2])
                        all_expansion_probs.append(exp_prob)
        
        if all_survival_probs:
            avg_survival = float(np.mean(all_survival_probs))
        else:
            avg_survival = 0.0
        
        if all_expansion_probs:
            avg_expansion = float(np.mean(all_expansion_probs))
        else:
            avg_expansion = 0.0
        
        round_data[rn] = {
            "survival_rate": avg_survival,
            "expansion_rate": avg_expansion,
            "features": seed_features,
        }
        print(f"  Round {rn}: survival={avg_survival:.4f} expansion={avg_expansion:.4f} "
              f"({len(all_survival_probs)} sett, {len(all_expansion_probs)} plains, "
              f"{len(seed_features)} total cells)")
    
    # Group by context key, indexed by round
    context_groups = defaultdict(lambda: defaultdict(list))
    
    for rn, data in round_data.items():
        for features, gt_dist in data["features"]:
            key = _make_context_key(features)
            if key is None:
                continue
            context_groups[key][rn].append(np.array(gt_dist))
    
    # Build 2D-indexed KB
    survival_rates = {rn: data["survival_rate"] for rn, data in round_data.items()}
    expansion_rates = {rn: data["expansion_rate"] for rn, data in round_data.items()}
    
    kb = {
        "round_survival_rates": survival_rates,
        "round_expansion_rates": expansion_rates,
        "context_keys": {}
    }
    
    for key, round_dists in context_groups.items():
        s_rates = []
        e_rates = []
        dists = []
        counts = []
        
        for rn in sorted(round_dists.keys()):
            s_rate = survival_rates[rn]
            e_rate = expansion_rates[rn]
            avg_dist = np.mean(round_dists[rn], axis=0).tolist()
            n = len(round_dists[rn])
            s_rates.append(s_rate)
            e_rates.append(e_rate)
            dists.append(avg_dist)
            counts.append(n)
        
        kb["context_keys"][key] = {
            "survival_rates": s_rates,
            "expansion_rates": e_rates,
            "dists": dists,
            "counts": counts,
        }
    
    # Save
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    kb_path = os.path.join(KNOWLEDGE_DIR, "expansion_indexed_kb.json")
    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2)
    
    print(f"\nSaved expansion-indexed KB: {len(kb['context_keys'])} context keys")
    print(f"Survival rates: {survival_rates}")
    print(f"Expansion rates: {expansion_rates}")
    
    # Show rounds spread in 2D
    print(f"\n--- Round 2D coordinates ---")
    for rn in sorted(survival_rates.keys()):
        print(f"  R{rn}: survival={survival_rates[rn]:.3f} expansion={expansion_rates[rn]:.4f}")
    
    return kb


def _make_context_key(features):
    """Create context key from cell features (V3 compatible)."""
    cls = features["class"]
    raw = features["raw"]
    
    if features["is_port"]:
        coast = "coastal" if features["is_coastal"] else "inland"
        return f"port|{coast}|forest_{min(features['n_forest'], 3)}"
    elif features["is_settlement"]:
        coast = "coastal" if features["is_coastal"] else "inland"
        return f"settlement|{coast}|forest_{min(features['n_forest'], 3)}"
    elif cls == 4:
        dist = min(features["dist_to_nearest_settlement"], 10)
        if features["is_coastal"]:
            return f"forest|near_sett_{dist}|coastal"
        return f"forest|near_sett_{dist}"
    elif raw == 11:
        coast = "coastal" if features["is_coastal"] else "inland"
        dist = min(features["dist_to_nearest_settlement"], 10)
        return f"plains|near_sett_{dist}|{coast}"
    
    return None


if __name__ == "__main__":
    build_expansion_kb()
