"""
Analyze which features best predict per-cell GT distributions.
Find the features that reduce within-group variance the most.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from learn import extract_cell_features, KNOWLEDGE_DIR


def analyze_feature_importance():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    # Load all GT + features
    all_data = []  # (features_dict, gt_dist, round_survival)
    
    round_survivals = {}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        
        # Compute round survival first
        survival_probs = []
        for si in range(detail["seeds_count"]):
            try:
                analysis = client.get_analysis(round_id, si)
            except:
                continue
            gt = np.array(analysis["ground_truth"])
            settlements = detail["initial_states"][si]["settlements"]
            for s in settlements:
                if s["alive"]:
                    sp = float(gt[s["y"], s["x"], 1] + gt[s["y"], s["x"], 2])
                    survival_probs.append(sp)
        
        if survival_probs:
            round_survivals[rn] = np.mean(survival_probs)
        else:
            continue
        
        # Now load all cells
        for si in range(detail["seeds_count"]):
            try:
                analysis = client.get_analysis(round_id, si)
            except:
                continue
            
            gt = np.array(analysis["ground_truth"])
            initial_grid = np.array(
                analysis.get("initial_grid") or detail["initial_states"][si]["grid"]
            )
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][si]["settlements"]
            settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
            port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}
            
            for y in range(H):
                for x in range(W):
                    cls = int(class_map[y, x])
                    raw = int(initial_grid[y, x])
                    if cls == 5 or raw == 10:
                        continue
                    
                    features = extract_cell_features(
                        x, y, class_map, initial_grid, settlement_set, port_set, W, H
                    )
                    # Add extra features
                    features["settlements_within_r3"] = sum(
                        1 for (sx, sy) in settlement_set
                        if abs(x - sx) + abs(y - sy) <= 3
                    )
                    features["n_initial_settlements"] = len(settlement_set)
                    
                    all_data.append((features, gt[y, x], round_survivals[rn]))
    
    print(f"Loaded {len(all_data)} cells across {len(round_survivals)} rounds")
    print(f"Round survivals: {round_survivals}")
    
    # Group by current context key first
    print("\n=== VARIANCE REDUCTION WITH ADDITIONAL SPLITS ===")
    
    # For each terrain type, test if splitting on additional features reduces KL
    for terrain_type in ["settlement", "plains_near", "plains_far", "forest_near", "forest_far"]:
        cells = []
        for feat, gt, surv in all_data:
            if terrain_type == "settlement" and (feat["is_settlement"] or feat["is_port"]):
                cells.append((feat, gt, surv))
            elif terrain_type == "plains_near" and feat["raw"] == 11 and feat["dist_to_nearest_settlement"] <= 3:
                cells.append((feat, gt, surv))
            elif terrain_type == "plains_far" and feat["raw"] == 11 and feat["dist_to_nearest_settlement"] > 3:
                cells.append((feat, gt, surv))
            elif terrain_type == "forest_near" and feat["class"] == 4 and feat["dist_to_nearest_settlement"] <= 3:
                cells.append((feat, gt, surv))
            elif terrain_type == "forest_far" and feat["class"] == 4 and feat["dist_to_nearest_settlement"] > 3:
                cells.append((feat, gt, surv))
        
        if len(cells) < 20:
            continue
        
        # Compute overall avg distribution
        all_gt = np.array([c[1] for c in cells])
        overall_avg = np.mean(all_gt, axis=0)
        overall_var = np.mean(np.sum((all_gt - overall_avg)**2, axis=1))
        
        print(f"\n--- {terrain_type} (n={len(cells)}) ---")
        print(f"  Overall variance: {overall_var:.6f}")
        
        # Test various split features
        for split_feat, split_vals in [
            ("dist_to_nearest_settlement", [0, 1, 2, 3, 4, 5]),
            ("settlements_within_r5", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            ("settlements_within_r3", [0, 1, 2, 3]),
            ("n_forest", [0, 1, 2, 3]),
            ("is_coastal", [False, True]),
            ("is_edge", [False, True]),
            ("n_settlement_neighbors", [0, 1, 2, 3]),
            ("n_initial_settlements", None),  # Will bin
        ]:
            groups = defaultdict(list)
            
            for feat, gt, surv in cells:
                val = feat.get(split_feat, 0)
                if split_vals is None:
                    # Bin numeric values
                    if isinstance(val, (int, float)):
                        val = int(val) // 10 * 10  # bin by 10s
                groups[val].append(gt)
            
            if len(groups) < 2:
                continue
            
            # Compute within-group variance
            weighted_var = 0
            total_n = 0
            for val, gts in groups.items():
                gts = np.array(gts)
                n = len(gts)
                avg = np.mean(gts, axis=0)
                var = np.mean(np.sum((gts - avg)**2, axis=1))
                weighted_var += var * n
                total_n += n
            
            within_var = weighted_var / max(total_n, 1)
            reduction = 100 * (1 - within_var / max(overall_var, 1e-10))
            
            n_groups_used = sum(1 for g in groups.values() if len(g) >= 5)
            print(f"  Split by {split_feat:30s}: within_var={within_var:.6f} "
                  f"reduction={reduction:.1f}% ({n_groups_used} groups with n>=5)")
    
    # Special analysis: does settlements_within_r3 help for plains?
    print("\n=== SETTLEMENTS_WITHIN_R3 SPLITS FOR PLAINS DIST 1-3 ===")
    for dist in [1, 2, 3]:
        cells_by_density = defaultdict(list)
        for feat, gt, surv in all_data:
            if feat["raw"] == 11 and feat["dist_to_nearest_settlement"] == dist and not feat["is_coastal"]:
                density = min(feat["settlements_within_r3"], 3)
                cells_by_density[density].append(gt)
        
        if not cells_by_density:
            continue
        
        print(f"\n  Plains dist={dist}:")
        for density in sorted(cells_by_density.keys()):
            gts = np.array(cells_by_density[density])
            avg = np.mean(gts, axis=0)
            n = len(gts)
            if n >= 5:
                print(f"    sett_within_r3={density} (n={n}): "
                      f"{' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.3f}' for i in range(6))}")
    
    # Settlement-specific: does n_settlement_neighbors matter?
    print("\n=== SETTLEMENT NEIGHBORS SPLIT FOR SETTLEMENTS ===")
    for n_neighbors in range(4):
        cells = [gt for feat, gt, surv in all_data 
                 if feat["is_settlement"] and feat["n_settlement_neighbors"] == n_neighbors]
        if len(cells) >= 10:
            avg = np.mean(cells, axis=0)
            print(f"  n_sett_neighbors={n_neighbors} (n={len(cells)}): "
                  f"{' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.3f}' for i in range(6))}")


if __name__ == "__main__":
    analyze_feature_importance()
