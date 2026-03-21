"""
Deep analysis: Find the second hidden parameter.

R7 and R2 have nearly identical survival rates (~0.42) but very different
oracle scores (70 vs 88). Something else is varying between rounds.

Candidate hidden parameters from simulation mechanics:
1. Expansion rate (how aggressively settlements found new ones)
2. Conflict intensity (how much raiding happens)
3. Winter severity (how often settlements starve/collapse)
4. Trade range/effectiveness
5. Reclamation speed (forest regrowth on ruins)

We analyze GT data across all rounds to find which metrics explain the
residual variance after survival rate is accounted for.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from learn import KNOWLEDGE_DIR, extract_cell_features
from model_v3 import load_survival_kb, interpolate_distribution, _make_context_key


def analyze_all_rounds():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    completed.sort(key=lambda r: r["round_number"])
    
    print(f"Analyzing {len(completed)} completed rounds for hidden parameters\n")
    
    round_metrics = {}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        # Aggregated metrics across all seeds
        all_survival_probs = []
        all_expansion_counts = []
        all_ruin_probs = []
        all_port_probs = []
        all_cell_entropies = []
        all_forest_change_probs = []  # How much forest changes
        all_empty_to_sett_probs = []  # Expansion: empty -> settlement
        all_sett_to_ruin_probs = []   # Collapse: settlement -> ruin
        all_plains_dists = defaultdict(list)  # by distance to settlement
        all_sett_dists = []
        all_gt_class_fracs = []  # Per-seed: fraction of cells in each class
        
        for seed_idx in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, seed_idx)
            except Exception as e:
                print(f"  R{rn} seed {seed_idx}: FAILED ({e})")
                continue
            
            gt = np.array(analysis["ground_truth"])  # H x W x 6
            initial_grid = np.array(
                analysis.get("initial_grid") or detail["initial_states"][seed_idx]["grid"]
            )
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][seed_idx]["settlements"]
            settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
            port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}
            
            # Per-cell GT analysis
            seed_entropies = []
            seed_expansion = 0
            seed_total_non_sett = 0
            
            for y in range(H):
                for x in range(W):
                    cls = int(class_map[y, x])
                    raw = int(initial_grid[y, x])
                    gt_dist = gt[y, x]
                    
                    # Skip static cells
                    if cls == 5 or raw == 10:
                        continue
                    
                    # Entropy
                    entropy = -np.sum(gt_dist * np.log(gt_dist + 1e-10))
                    if entropy > 0.05:
                        seed_entropies.append(entropy)
                    
                    # Settlement survival
                    if (x, y) in settlement_set:
                        surv_prob = float(gt_dist[1] + gt_dist[2])
                        all_survival_probs.append(surv_prob)
                        all_ruin_probs.append(float(gt_dist[3]))
                        all_sett_dists.append(gt_dist.tolist())
                    
                    # Port probability for settlements
                    if (x, y) in settlement_set:
                        all_port_probs.append(float(gt_dist[2]))
                    
                    # Expansion: non-settlement cells becoming settlement
                    if (x, y) not in settlement_set:
                        new_sett_prob = float(gt_dist[1] + gt_dist[2])
                        if raw == 11:  # Plains
                            all_empty_to_sett_probs.append(new_sett_prob)
                            seed_expansion += new_sett_prob
                            seed_total_non_sett += 1
                            
                            # Distance to nearest settlement
                            min_d = min(
                                (abs(x - sx) + abs(y - sy) for sx, sy in settlement_set),
                                default=999
                            )
                            all_plains_dists[min(min_d, 10)].append(gt_dist.tolist())
                        
                    # Forest change
                    if cls == 4:
                        forest_stay = float(gt_dist[4])
                        all_forest_change_probs.append(1.0 - forest_stay)
                    
                    # Settlement -> ruin
                    if (x, y) in settlement_set:
                        all_sett_to_ruin_probs.append(float(gt_dist[3]))
            
            all_cell_entropies.extend(seed_entropies)
            if seed_total_non_sett > 0:
                all_expansion_counts.append(seed_expansion / seed_total_non_sett)
            
            # Overall class fractions from GT
            # Expected class distribution
            gt_class_frac = np.zeros(6)
            for c in range(6):
                gt_class_frac[c] = np.mean(gt[:, :, c])
            all_gt_class_fracs.append(gt_class_frac)
        
        # Compute round-level metrics
        if not all_survival_probs:
            continue
        
        avg_survival = np.mean(all_survival_probs)
        std_survival = np.std(all_survival_probs)
        avg_ruin = np.mean(all_ruin_probs) if all_ruin_probs else 0
        avg_port = np.mean(all_port_probs) if all_port_probs else 0
        avg_entropy = np.mean(all_cell_entropies) if all_cell_entropies else 0
        avg_expansion = np.mean(all_expansion_counts) if all_expansion_counts else 0
        avg_forest_change = np.mean(all_forest_change_probs) if all_forest_change_probs else 0
        avg_empty_to_sett = np.mean(all_empty_to_sett_probs) if all_empty_to_sett_probs else 0
        avg_sett_to_ruin = np.mean(all_sett_to_ruin_probs) if all_sett_to_ruin_probs else 0
        
        # Collapse vs disappear ratio: ruin prob / (1 - survival prob)
        death_prob = 1 - avg_survival
        ruin_given_death = avg_sett_to_ruin / max(death_prob, 0.01)
        
        # Expansion by distance
        exp_by_dist = {}
        for d in sorted(all_plains_dists.keys()):
            dists = np.array(all_plains_dists[d])
            exp_by_dist[d] = float(np.mean(dists[:, 1] + dists[:, 2]))
        
        # Average GT class fracs
        avg_class_fracs = np.mean(all_gt_class_fracs, axis=0) if all_gt_class_fracs else np.zeros(6)
        
        round_metrics[rn] = {
            "survival": avg_survival,
            "survival_std": std_survival,
            "ruin_rate": avg_ruin,
            "port_prob": avg_port,
            "avg_entropy": avg_entropy,
            "expansion_rate": avg_expansion,
            "forest_change": avg_forest_change,
            "empty_to_sett": avg_empty_to_sett,
            "sett_to_ruin": avg_sett_to_ruin,
            "ruin_given_death": ruin_given_death,
            "exp_by_dist": exp_by_dist,
            "class_fracs": avg_class_fracs.tolist(),
            "n_settlements": len(all_survival_probs),
        }
        
        print(f"R{rn:2d}: surv={avg_survival:.3f}±{std_survival:.3f}  "
              f"ruin={avg_ruin:.3f}  exp={avg_expansion:.4f}  "
              f"forest_chg={avg_forest_change:.3f}  entropy={avg_entropy:.3f}  "
              f"ruin|death={ruin_given_death:.3f}  "
              f"empty->sett={avg_empty_to_sett:.4f}")
    
    # ════════════════════════════════════════════════════════════════
    # Analysis: Which metrics vary independently of survival rate?
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    rounds = sorted(round_metrics.keys())
    survival = np.array([round_metrics[r]["survival"] for r in rounds])
    
    metrics = {
        "ruin_rate": np.array([round_metrics[r]["ruin_rate"] for r in rounds]),
        "expansion_rate": np.array([round_metrics[r]["expansion_rate"] for r in rounds]),
        "forest_change": np.array([round_metrics[r]["forest_change"] for r in rounds]),
        "avg_entropy": np.array([round_metrics[r]["avg_entropy"] for r in rounds]),
        "ruin_given_death": np.array([round_metrics[r]["ruin_given_death"] for r in rounds]),
        "survival_std": np.array([round_metrics[r]["survival_std"] for r in rounds]),
        "port_prob": np.array([round_metrics[r]["port_prob"] for r in rounds]),
        "empty_to_sett": np.array([round_metrics[r]["empty_to_sett"] for r in rounds]),
    }
    
    print(f"\nCorrelation with survival rate (R = high means redundant with survival):")
    for name, vals in metrics.items():
        if np.std(vals) < 1e-10 or np.std(survival) < 1e-10:
            continue
        corr = np.corrcoef(survival, vals)[0, 1]
        print(f"  {name:20s}: R = {corr:+.3f}  (range: {np.min(vals):.4f} - {np.max(vals):.4f})")
    
    # Residual variance after accounting for survival
    print(f"\n{'='*80}")
    print("RESIDUAL ANALYSIS: Which rounds are most anomalous?")
    print(f"{'='*80}")
    
    # For each metric, compute residual after linear fit to survival
    for name, vals in metrics.items():
        if np.std(vals) < 1e-10:
            continue
        # Linear fit: metric = a * survival + b
        A = np.vstack([survival, np.ones_like(survival)]).T
        a, b = np.linalg.lstsq(A, vals, rcond=None)[0]
        predicted = a * survival + b
        residual = vals - predicted
        
        if np.std(residual) > 0.001:
            print(f"\n  {name} (residual std = {np.std(residual):.4f}):")
            for i, rn in enumerate(rounds):
                if abs(residual[i]) > np.std(residual) * 0.5:
                    print(f"    R{rn:2d}: actual={vals[i]:.4f} predicted={predicted[i]:.4f} "
                          f"residual={residual[i]:+.4f} (surv={survival[i]:.3f})")
    
    # ════════════════════════════════════════════════════════════════
    # Focus: R7 vs R2 (same survival, different oracle scores)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print("DEEP DIVE: Comparing rounds with similar survival rates")
    print(f"{'='*80}")
    
    # Group rounds by survival rate
    for rn in rounds:
        m = round_metrics[rn]
        print(f"\nR{rn:2d} (surv={m['survival']:.3f}):")
        print(f"  ruin_rate={m['ruin_rate']:.4f}  expansion={m['expansion_rate']:.5f}")
        print(f"  forest_change={m['forest_change']:.4f}  entropy={m['avg_entropy']:.4f}")
        print(f"  ruin|death={m['ruin_given_death']:.4f}  survival_std={m['survival_std']:.4f}")
        print(f"  class_fracs: {' '.join(f'{CLASS_NAMES[c]}:{m['class_fracs'][c]:.3f}' for c in range(6))}")
        print(f"  expansion by distance:")
        for d in sorted(m["exp_by_dist"].keys()):
            print(f"    d={d}: {m['exp_by_dist'][d]:.4f}")
    
    # ════════════════════════════════════════════════════════════════
    # Test: How much would a 2D-indexed KB help?
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print("SECOND PARAMETER CANDIDATES")
    print(f"{'='*80}")
    
    # For each candidate second parameter, compute how well a 2D lookup would work
    # by measuring within-group variance when rounds are matched on both survival + second param
    
    survival_kb = load_survival_kb()
    if survival_kb is None:
        print("No KB found!")
        return round_metrics
    
    context_keys = survival_kb.get("context_keys", {})
    
    # Compute oracle scores using just survival vs survival + candidate param
    for name, vals in metrics.items():
        if np.std(vals) < 1e-10:
            continue
        corr_with_surv = abs(np.corrcoef(survival, vals)[0, 1])
        if corr_with_surv > 0.95:
            print(f"  {name:20s}: Too correlated with survival (R={corr_with_surv:.3f}), skipping")
            continue
        
        # For each round, find the closest round in 2D space (survival, metric)
        # and compute the prediction quality improvement
        print(f"\n  Candidate: {name} (corr with survival = {corr_with_surv:.3f})")
        print(f"    Values: {', '.join(f'R{rn}={vals[i]:.4f}' for i, rn in enumerate(rounds))}")
    
    # Save metrics for use in model
    metrics_path = os.path.join(KNOWLEDGE_DIR, "round_metrics.json")
    serializable = {}
    for rn, m in round_metrics.items():
        serializable[str(rn)] = {k: v if isinstance(v, (int, float, str, list)) else 
                                    {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                                 for k, v in m.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved round metrics to {metrics_path}")
    
    return round_metrics


if __name__ == "__main__":
    analyze_all_rounds()
