"""
V4 vs V5 Comparison — LOO-style validation across all completed rounds.

Tests both models on every round:
  - V4 MLE: V4 model with MLE-estimated parameters (what we actually submit)
  - V4 Oracle: V4 model with ground-truth parameters 
  - V5 MLE: V5 model with MLE-estimated parameters
  - V5 Oracle: V5 model with ground-truth parameters

For rounds with saved observations, we also test the Bayesian update path
(V5's main improvement: settlement health-weighted soft observations).
"""

import numpy as np
import json
import os
import sys
import io
from client import AstarClient, grid_to_class_map
from config import PROB_FLOOR, NUM_CLASSES, CLASS_NAMES, TERRAIN_TO_CLASS

from model_v4 import (
    build_prediction_v4, load_survival_kb, load_expansion_kb,
    estimate_survival_mle, estimate_expansion_from_obs,
    DEFAULT_SURVIVAL_RATE, DEFAULT_EXPANSION_RATE,
)
from model_v5 import build_prediction_v5


def compute_score(gt, pred):
    """Compute entropy-weighted KL divergence score (same as server)."""
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


def validate_v4_v5():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    if not survival_kb:
        print("ERROR: No survival KB.")
        return
    
    print(f"\n{'='*90}")
    print("V4 vs V5 COMPARISON — All Completed Rounds")
    print(f"{'='*90}")
    print(f"{'Round':>6s} {'Obs':>4s} {'GTsurv':>7s} {'MLE_s':>7s} | "
          f"{'V4_orc':>7s} {'V4_MLE':>7s} {'V5_orc':>7s} {'V5_MLE':>7s} | "
          f"{'Δorc':>6s} {'ΔMLE':>6s} | {'Submit':>7s}")
    print("-" * 90)
    
    results = []
    
    for r in sorted(completed, key=lambda x: x["round_number"]):
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        # Load observations if available
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
        
        # Ground truth survival + expansion rates
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
                for y_cell in range(H):
                    for x_cell in range(W):
                        if int(initial_grid[y_cell, x_cell]) == 11 and (x_cell, y_cell) not in settlement_set:
                            gt_expansions.append(float(gt_data[y_cell, x_cell, 1] + gt_data[y_cell, x_cell, 2]))
            except:
                pass
        
        gt_survival = np.mean(gt_survivals) if gt_survivals else 0.28
        gt_expansion = np.mean(gt_expansions) if gt_expansions else 0.13
        
        # MLE estimates
        if has_obs:
            # Suppress MLE prints
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            est_survival, _ = estimate_survival_mle(observations, all_seeds_initial_states, W, H, survival_kb)
            est_expansion, _ = estimate_expansion_from_obs(observations, all_seeds_initial_states, W, H)
            sys.stdout = old_stdout
        else:
            est_survival = DEFAULT_SURVIVAL_RATE
            est_expansion = DEFAULT_EXPANSION_RATE
        
        # Score each model variant
        v4_oracle_scores = []
        v4_mle_scores = []
        v5_oracle_scores = []
        v5_mle_scores = []
        
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
            except:
                continue
            
            gt = np.array(analysis["ground_truth"])
            initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][si]["grid"])
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][si]["settlements"]
            
            # Suppress model prints
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # V4 Oracle
                pred = build_prediction_v4(
                    class_map, initial_grid, settlements,
                    observations, si, W, H,
                    survival_kb=survival_kb, expansion_kb=expansion_kb,
                    estimated_survival=gt_survival,
                    estimated_expansion=gt_expansion,
                    all_seeds_settlements=all_seeds_settlements,
                )
                sc, _, _ = compute_score(gt, pred)
                v4_oracle_scores.append(sc)
                
                # V4 MLE
                pred = build_prediction_v4(
                    class_map, initial_grid, settlements,
                    observations, si, W, H,
                    survival_kb=survival_kb, expansion_kb=expansion_kb,
                    estimated_survival=est_survival,
                    estimated_expansion=est_expansion,
                    all_seeds_settlements=all_seeds_settlements,
                )
                sc, _, _ = compute_score(gt, pred)
                v4_mle_scores.append(sc)
                
                # V5 Oracle
                pred = build_prediction_v5(
                    class_map, initial_grid, settlements,
                    observations, si, W, H,
                    survival_kb=survival_kb, expansion_kb=expansion_kb,
                    estimated_survival=gt_survival,
                    estimated_expansion=gt_expansion,
                    all_seeds_settlements=all_seeds_settlements,
                )
                sc, _, _ = compute_score(gt, pred)
                v5_oracle_scores.append(sc)
                
                # V5 MLE
                pred = build_prediction_v5(
                    class_map, initial_grid, settlements,
                    observations, si, W, H,
                    survival_kb=survival_kb, expansion_kb=expansion_kb,
                    estimated_survival=est_survival,
                    estimated_expansion=est_expansion,
                    all_seeds_settlements=all_seeds_settlements,
                )
                sc, _, _ = compute_score(gt, pred)
                v5_mle_scores.append(sc)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"  ERROR on R{rn} seed {si}: {e}")
                continue
            finally:
                sys.stdout = old_stdout
        
        # Average scores
        avg_v4_orc = np.mean(v4_oracle_scores) if v4_oracle_scores else 0
        avg_v4_mle = np.mean(v4_mle_scores) if v4_mle_scores else 0
        avg_v5_orc = np.mean(v5_oracle_scores) if v5_oracle_scores else 0
        avg_v5_mle = np.mean(v5_mle_scores) if v5_mle_scores else 0
        
        delta_orc = avg_v5_orc - avg_v4_orc
        delta_mle = avg_v5_mle - avg_v4_mle
        
        submitted = r.get("round_score") or 0
        obs_str = "YES" if has_obs else " no"
        
        print(f"  R{rn:2d} ({obs_str}) surv={gt_survival:.3f} MLE={est_survival:.3f} | "
              f"V4o={avg_v4_orc:.1f} V4m={avg_v4_mle:.1f} "
              f"V5o={avg_v5_orc:.1f} V5m={avg_v5_mle:.1f} | "
              f"Δorc={delta_orc:+.2f} ΔMLE={delta_mle:+.2f} | sub={submitted:.1f}")
        
        results.append({
            "rn": rn, "has_obs": has_obs,
            "gt_surv": gt_survival, "gt_exp": gt_expansion,
            "est_surv": est_survival, "est_exp": est_expansion,
            "v4_orc": avg_v4_orc, "v4_mle": avg_v4_mle,
            "v5_orc": avg_v5_orc, "v5_mle": avg_v5_mle,
            "submitted": submitted,
            "weight": r.get("round_weight", 1.0) or 1.0,
        })
    
    # ── Summary ──
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    
    all_v4_orc = [r["v4_orc"] for r in results]
    all_v4_mle = [r["v4_mle"] for r in results]
    all_v5_orc = [r["v5_orc"] for r in results]
    all_v5_mle = [r["v5_mle"] for r in results]
    
    print(f"\nAll {len(results)} rounds:")
    print(f"  V4 Oracle:  {np.mean(all_v4_orc):.2f}")
    print(f"  V4 MLE:     {np.mean(all_v4_mle):.2f}")
    print(f"  V5 Oracle:  {np.mean(all_v5_orc):.2f} ({np.mean(all_v5_orc) - np.mean(all_v4_orc):+.2f} vs V4)")
    print(f"  V5 MLE:     {np.mean(all_v5_mle):.2f} ({np.mean(all_v5_mle) - np.mean(all_v4_mle):+.2f} vs V4)")
    
    # Rounds WITH observations (where V5 health scoring matters)
    obs_results = [r for r in results if r["has_obs"]]
    if obs_results:
        print(f"\n{len(obs_results)} rounds WITH observations:")
        obs_v4_orc = [r["v4_orc"] for r in obs_results]
        obs_v4_mle = [r["v4_mle"] for r in obs_results]
        obs_v5_orc = [r["v5_orc"] for r in obs_results]
        obs_v5_mle = [r["v5_mle"] for r in obs_results]
        print(f"  V4 Oracle:  {np.mean(obs_v4_orc):.2f}")
        print(f"  V4 MLE:     {np.mean(obs_v4_mle):.2f}")
        print(f"  V5 Oracle:  {np.mean(obs_v5_orc):.2f} ({np.mean(obs_v5_orc) - np.mean(obs_v4_orc):+.2f} vs V4)")
        print(f"  V5 MLE:     {np.mean(obs_v5_mle):.2f} ({np.mean(obs_v5_mle) - np.mean(obs_v4_mle):+.2f} vs V4)")
    
    # Rounds WITHOUT observations
    noobs = [r for r in results if not r["has_obs"]]
    if noobs:
        print(f"\n{len(noobs)} rounds WITHOUT observations:")
        noobs_v4_orc = [r["v4_orc"] for r in noobs]
        noobs_v5_orc = [r["v5_orc"] for r in noobs]
        print(f"  V4 Oracle:  {np.mean(noobs_v4_orc):.2f}")
        print(f"  V5 Oracle:  {np.mean(noobs_v5_orc):.2f} ({np.mean(noobs_v5_orc) - np.mean(noobs_v4_orc):+.2f} vs V4)")
    
    # Per-round delta breakdown
    print(f"\nPer-round V5-V4 delta (MLE):")
    obs_deltas = []
    for r in results:
        delta = r["v5_mle"] - r["v4_mle"]
        marker = "★" if r["has_obs"] else " "
        print(f"  {marker} R{r['rn']:2d}: {delta:+.3f}")
        if r["has_obs"]:
            obs_deltas.append(delta)
    
    if obs_deltas:
        print(f"\n  Avg delta (obs rounds): {np.mean(obs_deltas):+.3f}")
        print(f"  Win/Loss: {sum(1 for d in obs_deltas if d > 0)}/{sum(1 for d in obs_deltas if d < 0)}")
    
    # Best potential weighted scores  
    print(f"\nWeighted score comparison (top 5 rounds by weight):")
    by_weight = sorted(results, key=lambda r: r["weight"], reverse=True)[:5]
    for r in by_weight:
        v4w = r["v4_mle"] * r["weight"]
        v5w = r["v5_mle"] * r["weight"]
        print(f"  R{r['rn']:2d} (w={r['weight']:.2f}): V4={v4w:.1f}  V5={v5w:.1f}  Δ={v5w-v4w:+.1f}")


if __name__ == "__main__":
    validate_v4_v5()
