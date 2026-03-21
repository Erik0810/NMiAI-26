"""
Validation script for V4 model — focused comparison.
Tests: V3 oracle, V4 adaptive Bayesian, V4 + 2D KB, and V4 full.
"""

import numpy as np
import json
import os
from client import AstarClient, grid_to_class_map
from config import PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model_v4 import (
    build_prediction_v4, load_survival_kb, load_expansion_kb,
    estimate_survival_mle, estimate_expansion_from_obs,
    DEFAULT_SURVIVAL_RATE, DEFAULT_EXPANSION_RATE,
)
# Import original V3 for comparison
import model_v3 as v3_module


def compute_score(gt, pred):
    gt_safe = np.maximum(gt, 1e-10)
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)
    entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    dynamic_mask = entropy > 0.01
    if not np.any(dynamic_mask):
        return 100.0, 0.0, 0
    weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / np.sum(entropy[dynamic_mask])
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score, weighted_kl, int(np.sum(dynamic_mask))


def validate():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    if not survival_kb:
        print("ERROR: No survival KB.")
        return
    
    print(f"\n{'='*80}")
    print("V4 VALIDATION — Adaptive Bayesian + 2D KB")
    print(f"{'='*80}")
    
    results = []
    
    for r in sorted(completed, key=lambda x: x["round_number"]):
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        obs_path = f"round_data/observations_{round_id}.json"
        if os.path.exists(obs_path):
            with open(obs_path) as f:
                observations = json.load(f)
            has_obs = True
        else:
            observations = []
            has_obs = False
        
        all_seeds_initial_states = {si: detail["initial_states"][si] for si in range(seeds_count)}
        all_seeds_settlements = {si: detail["initial_states"][si]["settlements"] for si in range(seeds_count)}
        
        # GT values
        gt_survivals = []
        gt_expansions = []
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
                gt_data = np.array(analysis["ground_truth"])
                settlements = all_seeds_settlements[si]
                settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
                for (sx, sy) in settlement_set:
                    gt_survivals.append(float(gt_data[sy, sx, 1] + gt_data[sy, sx, 2]))
                initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][si]["grid"])
                for y in range(H):
                    for x in range(W):
                        if int(initial_grid[y, x]) == 11 and (x, y) not in settlement_set:
                            gt_expansions.append(float(gt_data[y, x, 1] + gt_data[y, x, 2]))
            except:
                pass
        
        gt_survival = np.mean(gt_survivals) if gt_survivals else 0.28
        gt_expansion = np.mean(gt_expansions) if gt_expansions else 0.13
        
        if has_obs:
            est_survival, _ = estimate_survival_mle(observations, all_seeds_initial_states, W, H, survival_kb)
            est_expansion, _ = estimate_expansion_from_obs(observations, all_seeds_initial_states, W, H)
        else:
            est_survival = DEFAULT_SURVIVAL_RATE
            est_expansion = DEFAULT_EXPANSION_RATE
        
        # Score each config (suppress verbose output)
        import io, sys
        old_stdout = sys.stdout
        
        v3_scores = []
        v4_bayes_scores = []
        v4_2d_scores = []
        v4_full_scores = []
        v4_mle_scores = []
        
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
            except:
                continue
            
            gt = np.array(analysis["ground_truth"])
            initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][si]["grid"])
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][si]["settlements"]
            
            # Suppress prints from model
            sys.stdout = io.StringIO()
            
            # V3 oracle
            pred_v3 = v3_module.build_prediction_v3(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb,
                estimated_survival=gt_survival,
                all_seeds_settlements=all_seeds_settlements,
            )
            sc, _, _ = compute_score(gt, pred_v3)
            v3_scores.append(sc)
            
            # V4 adaptive Bayesian only (1D KB, oracle survival)
            pred_v4b = build_prediction_v4(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb, expansion_kb=None,
                estimated_survival=gt_survival,
                estimated_expansion=gt_expansion,
                all_seeds_settlements=all_seeds_settlements,
            )
            sc, _, _ = compute_score(gt, pred_v4b)
            v4_bayes_scores.append(sc)
            
            # V4 2D KB without Bayesian (empty obs → no Bayesian)
            pred_v4_2d = build_prediction_v4(
                class_map, initial_grid, settlements,
                [], si, W, H,
                survival_kb=survival_kb, expansion_kb=expansion_kb,
                estimated_survival=gt_survival,
                estimated_expansion=gt_expansion,
                all_seeds_settlements=all_seeds_settlements,
            )
            sc, _, _ = compute_score(gt, pred_v4_2d)
            v4_2d_scores.append(sc)
            
            # V4 full (2D + Bayesian, oracle params)
            pred_v4f = build_prediction_v4(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb, expansion_kb=expansion_kb,
                estimated_survival=gt_survival,
                estimated_expansion=gt_expansion,
                all_seeds_settlements=all_seeds_settlements,
            )
            sc, _, _ = compute_score(gt, pred_v4f)
            v4_full_scores.append(sc)
            
            # V4 full with MLE params (what we'd actually submit)
            pred_v4m = build_prediction_v4(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb, expansion_kb=expansion_kb,
                estimated_survival=est_survival,
                estimated_expansion=est_expansion,
                all_seeds_settlements=all_seeds_settlements,
            )
            sc, _, _ = compute_score(gt, pred_v4m)
            v4_mle_scores.append(sc)
            
            sys.stdout = old_stdout
        
        sys.stdout = old_stdout
        
        avg_v3 = np.mean(v3_scores) if v3_scores else 0
        avg_bayes = np.mean(v4_bayes_scores) if v4_bayes_scores else 0
        avg_2d = np.mean(v4_2d_scores) if v4_2d_scores else 0
        avg_full = np.mean(v4_full_scores) if v4_full_scores else 0
        avg_mle = np.mean(v4_mle_scores) if v4_mle_scores else 0
        
        obs_str = "YES" if has_obs else " no"
        print(f"R{rn:2d} ({obs_str}): V3={avg_v3:.1f}  Bay={avg_bayes:.1f}({avg_bayes-avg_v3:+.1f})  "
              f"2D={avg_2d:.1f}({avg_2d-avg_v3:+.1f})  Full={avg_full:.1f}({avg_full-avg_v3:+.1f})  "
              f"MLE={avg_mle:.1f}({avg_mle-avg_v3:+.1f})")
        
        results.append({
            "rn": rn, "has_obs": has_obs,
            "gt_surv": gt_survival, "gt_exp": gt_expansion,
            "est_surv": est_survival, "est_exp": est_expansion,
            "v3": avg_v3, "bayes": avg_bayes, "2d": avg_2d,
            "full": avg_full, "mle": avg_mle,
            "submitted": r.get("round_score") or 0,
            "weight": r.get("round_weight", 1.0) or 1.0,
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    keys = ["v3", "bayes", "2d", "full", "mle"]
    labels = ["V3 oracle", "V4 Bayes+1D", "V4 2D(noObs)", "V4 Full(orc)", "V4 MLE"]
    
    print(f"\nAll {len(results)} rounds:")
    avg_v3 = np.mean([r["v3"] for r in results])
    for key, label in zip(keys, labels):
        avg = np.mean([r[key] for r in results])
        print(f"  {label:15s}: {avg:.2f} ({avg-avg_v3:+.2f})")
    
    obs_results = [r for r in results if r["has_obs"]]
    if obs_results:
        print(f"\n{len(obs_results)} rounds WITH observations:")
        avg_v3_obs = np.mean([r["v3"] for r in obs_results])
        for key, label in zip(keys, labels):
            avg = np.mean([r[key] for r in obs_results])
            print(f"  {label:15s}: {avg:.2f} ({avg-avg_v3_obs:+.2f})")
    
    noobs = [r for r in results if not r["has_obs"]]
    if noobs:
        print(f"\n{len(noobs)} rounds WITHOUT observations:")
        avg_v3_no = np.mean([r["v3"] for r in noobs])
        for key, label in zip(keys, labels):
            avg = np.mean([r[key] for r in noobs])
            print(f"  {label:15s}: {avg:.2f} ({avg-avg_v3_no:+.2f})")
    
    # Best weighted scores
    print(f"\nBest weighted scores:")
    for key, label in zip(keys, labels):
        best = max(r[key] * r["weight"] for r in results)
        best_r = max(results, key=lambda r: r[key] * r["weight"])
        print(f"  {label:15s}: {best:.1f} (R{best_r['rn']})")


if __name__ == "__main__":
    validate()
