"""
Validation script for V3 model — tests against saved GT data from all rounds.
Computes what our score would have been with the survival-indexed interpolation model.
"""

import numpy as np
import json
import os
import sys
import glob
from collections import defaultdict
from client import grid_to_class_map, AstarClient
from config import PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model_v3 import build_prediction_v3, load_survival_kb, _estimate_survival_rate, estimate_survival_mle


def compute_kl_divergence(gt, pred):
    gt_safe = np.maximum(gt, 1e-10)
    pred_safe = np.maximum(pred, 1e-10)
    return np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)


def compute_entropy(gt):
    gt_safe = np.maximum(gt, 1e-10)
    return -np.sum(gt_safe * np.log(gt_safe), axis=-1)


def compute_score(gt, pred):
    kl = compute_kl_divergence(gt, pred)
    entropy = compute_entropy(gt)
    dynamic_mask = entropy > 0.01
    if not np.any(dynamic_mask):
        return 100.0, 0.0, 0
    weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / np.sum(entropy[dynamic_mask])
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score, weighted_kl, int(np.sum(dynamic_mask))


def validate_v3():
    """Validate V3 model against ALL rounds with saved GT data."""
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    survival_kb = load_survival_kb()
    if not survival_kb:
        print("ERROR: No survival-indexed KB found. Run build_kb.py first.")
        return
    
    print(f"\n{'='*70}")
    print("V3 VALIDATION — Survival-Indexed Interpolation")
    print(f"{'='*70}")
    
    all_round_scores = {}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        print(f"\n--- Round {rn} ---")
        
        # Load observations if available
        obs_path = f"round_data/observations_{round_id}.json"
        if os.path.exists(obs_path):
            with open(obs_path) as f:
                observations = json.load(f)
            print(f"  Loaded {len(observations)} observations")
        else:
            observations = []
            print(f"  No observations available")
        
        # Build all_seeds_settlements dict
        all_seeds_settlements = {}
        for si in range(seeds_count):
            all_seeds_settlements[si] = detail["initial_states"][si]["settlements"]
        
        # Estimate survival rate from observations (or use GT for validation)
        if observations:
            # Simple estimate
            est_survival_simple, _ = _estimate_survival_rate(
                observations, all_seeds_settlements, W, H
            )
            print(f"  Simple survival estimate: {est_survival_simple:.4f}")
            
            # MLE estimate (uses all cell observations)
            all_seeds_initial_states = {
                si: detail["initial_states"][si] for si in range(seeds_count)
            }
            est_survival_mle, _ = estimate_survival_mle(
                observations, all_seeds_initial_states, W, H, survival_kb
            )
            
            est_survival = est_survival_mle  # Use MLE as primary
        else:
            est_survival = None
        
        # For comparison, compute GT survival rate
        gt_survivals = []
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
                gt = np.array(analysis["ground_truth"])
                settlements = all_seeds_settlements[si]
                settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
                for (sx, sy) in settlement_set:
                    gt_survivals.append(float(gt[sy, sx, 1] + gt[sy, sx, 2]))
            except:
                pass
        
        if gt_survivals:
            gt_survival = np.mean(gt_survivals)
            print(f"  GT survival rate: {gt_survival:.4f}")
        
        seed_scores_v3 = []
        seed_scores_v3_gt = []  # Using GT survival rate (oracle)
        
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
            except Exception as e:
                print(f"  Seed {si}: FAILED ({e})")
                continue
            
            gt = np.array(analysis["ground_truth"])
            initial_grid = np.array(
                analysis.get("initial_grid") or detail["initial_states"][si]["grid"]
            )
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][si]["settlements"]
            
            # V3 prediction with estimated survival
            pred_v3 = build_prediction_v3(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb,
                estimated_survival=est_survival,
                all_seeds_settlements=all_seeds_settlements,
            )
            score_v3, wkl_v3, n_dyn = compute_score(gt, pred_v3)
            seed_scores_v3.append(score_v3)
            
            # V3 prediction with GT survival (oracle — best possible)
            if gt_survivals:
                pred_v3_gt = build_prediction_v3(
                    class_map, initial_grid, settlements,
                    observations, si, W, H,
                    survival_kb=survival_kb,
                    estimated_survival=gt_survival,
                    all_seeds_settlements=all_seeds_settlements,
                )
                score_v3_gt, wkl_gt, _ = compute_score(gt, pred_v3_gt)
                seed_scores_v3_gt.append(score_v3_gt)
            
            oracle_str = f"{seed_scores_v3_gt[-1]:.2f}" if seed_scores_v3_gt else "N/A"
            actual_score = r.get("round_score") or analysis.get("score") or 0
            print(f"  Seed {si}: V3={score_v3:.2f} (wkl={wkl_v3:.4f}) "
                  f"V3_oracle={oracle_str} actual={actual_score}")
        
        if seed_scores_v3:
            avg_v3 = np.mean(seed_scores_v3)
            avg_oracle = np.mean(seed_scores_v3_gt) if seed_scores_v3_gt else 0
            actual = r.get("round_score") or 0
            weight = r.get("round_weight", 1.0) or 1.0
            print(f"  ROUND {rn} AVG: V3={avg_v3:.2f} oracle={avg_oracle:.2f} "
                  f"actual={actual} weight={weight}")
            print(f"  WEIGHTED: V3={avg_v3*weight:.2f} oracle={avg_oracle*weight:.2f}")
            all_round_scores[rn] = {
                "v3": avg_v3, "oracle": avg_oracle, "actual": actual or 0,
                "weight": weight,
            }
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    best_weighted = 0
    for rn, scores in sorted(all_round_scores.items()):
        weighted = scores["v3"] * scores["weight"]
        best_weighted = max(best_weighted, weighted)
        print(f"  Round {rn}: V3={scores['v3']:.2f} oracle={scores['oracle']:.2f} "
              f"actual={scores['actual']:.1f} weight={scores['weight']:.3f} "
              f"weighted_v3={weighted:.2f}")
    print(f"\n  Best V3 leaderboard score: {best_weighted:.2f}")


if __name__ == "__main__":
    validate_v3()
