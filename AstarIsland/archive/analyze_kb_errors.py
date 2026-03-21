"""
Analyze per-key KB errors to identify which context keys contribute most
to the oracle gap (15 pts avg). This tells us where to focus improvements.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from model_v3 import interpolate_distribution, _make_context_key, load_survival_kb
from learn import KNOWLEDGE_DIR


def analyze_kb_errors():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    survival_kb = load_survival_kb()
    if not survival_kb:
        print("No KB found!")
        return
    
    context_keys_kb = survival_kb.get("context_keys", {})
    
    # For each context key, track: (predicted_dist, gt_dist, entropy, cell_count)
    key_errors = defaultdict(lambda: {"kl_sum": 0.0, "ent_sum": 0.0, "count": 0,
                                       "weighted_kl_sum": 0.0})
    
    # For leave-one-out: track per-round errors
    round_key_errors = defaultdict(lambda: defaultdict(lambda: {"kl_sum": 0.0, "ent_sum": 0.0, "count": 0}))
    
    for r in sorted(completed, key=lambda x: x["round_number"]):
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        # GT survival rate
        gt_survivals = []
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
                gt = np.array(analysis["ground_truth"])
                settlements = detail["initial_states"][si]["settlements"]
                settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
                for (sx, sy) in settlement_set:
                    gt_survivals.append(float(gt[sy, sx, 1] + gt[sy, sx, 2]))
            except:
                pass
        
        gt_survival = np.mean(gt_survivals) if gt_survivals else 0.28
        
        # Now compute per-cell KB error
        for si in range(seeds_count):
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
            raw_grid = np.array(initial_grid)
            
            for y in range(H):
                for x in range(W):
                    cls = int(class_map[y, x])
                    raw = int(raw_grid[y, x])
                    
                    if cls == 5 or raw == 10:
                        continue
                    
                    gt_dist = gt[y, x]
                    gt_safe = np.maximum(gt_dist, 1e-10)
                    entropy = -np.sum(gt_safe * np.log(gt_safe))
                    
                    if entropy < 0.01:
                        continue
                    
                    # Get KB prediction with oracle survival
                    key, is_coastal, min_dist, n_forest = _make_context_key(
                        cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
                    )
                    
                    if key and key in context_keys_kb:
                        entry = context_keys_kb[key]
                        pred_dist = interpolate_distribution(
                            entry["rates"], entry["dists"], gt_survival
                        )
                    else:
                        continue  # Skip non-KB cells
                    
                    pred_safe = np.maximum(pred_dist, 1e-10)
                    kl = np.sum(gt_safe * np.log(gt_safe / pred_safe))
                    
                    key_errors[key]["kl_sum"] += kl
                    key_errors[key]["ent_sum"] += entropy
                    key_errors[key]["count"] += 1
                    key_errors[key]["weighted_kl_sum"] += entropy * kl
                    
                    round_key_errors[rn][key]["kl_sum"] += kl
                    round_key_errors[rn][key]["ent_sum"] += entropy
                    round_key_errors[rn][key]["count"] += 1
        
        print(f"  R{rn} done")
    
    # ═══════════════════════════════════════════════════════════════
    # Summary: which keys contribute most to total weighted KL?
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*90}")
    print("Per-key contribution to oracle gap (sorted by total weighted KL)")
    print(f"{'='*90}")
    
    total_weighted_kl = sum(d["weighted_kl_sum"] for d in key_errors.values())
    total_ent = sum(d["ent_sum"] for d in key_errors.values())
    
    sorted_keys = sorted(key_errors.items(), key=lambda x: x[1]["weighted_kl_sum"], reverse=True)
    
    cumulative_pct = 0.0
    print(f"\n{'Key':40s} {'Cells':>6} {'AvgKL':>7} {'AvgEnt':>7} {'WtdKL%':>7} {'Cum%':>6}")
    for key, data in sorted_keys[:30]:
        avg_kl = data["kl_sum"] / max(data["count"], 1)
        avg_ent = data["ent_sum"] / max(data["count"], 1)
        pct = 100.0 * data["weighted_kl_sum"] / total_weighted_kl
        cumulative_pct += pct
        print(f"  {key:38s} {data['count']:6d} {avg_kl:7.4f} {avg_ent:7.3f} {pct:6.1f}% {cumulative_pct:5.1f}%")
    
    # ═══════════════════════════════════════════════════════════════
    # Per-round: which rounds have highest variance per key?
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*90}")
    print("Worst rounds per key (shows within-key variance across rounds)")
    print(f"{'='*90}")
    
    for key, _ in sorted_keys[:10]:
        print(f"\n  {key}:")
        round_kls = {}
        for rn in sorted(round_key_errors.keys()):
            if key in round_key_errors[rn]:
                rd = round_key_errors[rn][key]
                avg_kl = rd["kl_sum"] / max(rd["count"], 1)
                round_kls[rn] = avg_kl
        
        for rn in sorted(round_kls.keys(), key=lambda r: round_kls[r], reverse=True):
            print(f"    R{rn:2d}: avg_kl={round_kls[rn]:.4f} ({round_key_errors[rn][key]['count']} cells)")
    
    # ═══════════════════════════════════════════════════════════════
    # What would perfect per-key per-round lookup give?
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*90}")
    print("Theoretical gain from richer key scheme")
    print(f"{'='*90}")
    
    # Check if splitting by round gives better fit (i.e., within-key variance)
    total_within_var = 0.0
    total_between_var = 0.0
    
    for key, data in sorted_keys[:20]:
        if key not in context_keys_kb:
            continue
        
        # Between-round variance for this key: how much do GT dists vary across rounds?
        round_avg_dists = {}
        for rn in round_key_errors.keys():
            if key not in round_key_errors[rn]:
                continue
            # We'd need GT dists here — skip detailed analysis for now
            pass
    
    print(f"\nTotal weighted KL across all keys: {total_weighted_kl:.4f}")
    print(f"Total entropy: {total_ent:.1f}")
    print(f"Average weighted KL: {total_weighted_kl / total_ent:.6f}")
    overall_score = 100 * np.exp(-3 * total_weighted_kl / total_ent)
    print(f"Implied oracle score: {overall_score:.1f}")
    
    return key_errors


if __name__ == "__main__":
    analyze_kb_errors()
