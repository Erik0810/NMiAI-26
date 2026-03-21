"""
Validate model predictions against saved ground truth data from Round 1.
Uses the GT data to compute what our score WOULD HAVE BEEN with the new model.

This lets us test model changes offline before the next round.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import grid_to_class_map
from config import PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from model import build_prediction_v2, load_knowledge_base


def compute_kl_divergence(gt, pred):
    """Compute KL(gt || pred) per cell."""
    # Avoid log(0)
    gt_safe = np.maximum(gt, 1e-10)
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)
    return kl


def compute_entropy(gt):
    """Compute entropy per cell."""
    gt_safe = np.maximum(gt, 1e-10)
    return -np.sum(gt_safe * np.log(gt_safe), axis=-1)


def compute_score(gt, pred):
    """
    Compute the official score for a single seed.
    score = 100 * exp(-3 * weighted_kl)
    """
    kl = compute_kl_divergence(gt, pred)
    entropy = compute_entropy(gt)
    
    # Only dynamic cells contribute
    dynamic_mask = entropy > 0.01
    if not np.any(dynamic_mask):
        return 100.0, 0.0, 0
    
    weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / np.sum(entropy[dynamic_mask])
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    n_dynamic = int(np.sum(dynamic_mask))
    
    return score, weighted_kl, n_dynamic


def validate_model(round_detail_path="round_detail.json", 
                   observations_path=None,
                   use_knowledge=True,
                   verbose=True):
    """
    Validate model predictions against Round 1 ground truth.
    
    Args:
        round_detail_path: Path to saved round detail JSON
        observations_path: Path to saved observations (if None, tests with NO observations)
        use_knowledge: Whether to use knowledge base
        verbose: Print detailed analysis
    
    Returns:
        dict with scores per seed and overall
    """
    # Load round detail
    with open(round_detail_path) as f:
        detail = json.load(f)
    
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    
    # Load observations (if available)
    observations = []
    if observations_path and os.path.exists(observations_path):
        with open(observations_path) as f:
            observations = json.load(f)
        print(f"Loaded {len(observations)} observations")
    else:
        print("No observations — testing with priors only")
    
    # Load knowledge base
    knowledge = load_knowledge_base() if use_knowledge else None
    
    # Load ground truth
    gt_data = {}
    for seed_idx in range(seeds_count):
        gt_path = f"round_data/gt_seed{seed_idx}.npz"
        if os.path.exists(gt_path):
            data = np.load(gt_path)
            gt_data[seed_idx] = data["ground_truth"]
        else:
            print(f"WARNING: No GT for seed {seed_idx}")
    
    if not gt_data:
        print("No ground truth data found! Run analyze_gt.py first.")
        return None
    
    # Build predictions and compare
    results = {}
    total_score = 0
    
    for seed_idx in range(seeds_count):
        if seed_idx not in gt_data:
            continue
        
        gt = gt_data[seed_idx]
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])
        
        print(f"\n{'='*50}")
        print(f"SEED {seed_idx}")
        print(f"{'='*50}")
        
        # Build prediction
        pred = build_prediction_v2(
            class_map, raw_grid, state["settlements"],
            observations, seed_idx, W, H, knowledge
        )
        
        # Compute score
        score, weighted_kl, n_dynamic = compute_score(gt, pred)
        
        print(f"  Score: {score:.1f} (weighted KL: {weighted_kl:.4f}, {n_dynamic} dynamic cells)")
        
        if verbose:
            # Detailed analysis
            kl = compute_kl_divergence(gt, pred)
            entropy = compute_entropy(gt)
            
            # Class distribution comparison
            gt_argmax = np.argmax(gt, axis=-1)
            pred_argmax = np.argmax(pred, axis=-1)
            
            print(f"\n  Class distribution (argmax):")
            for c in range(NUM_CLASSES):
                gt_count = np.sum(gt_argmax == c)
                pred_count = np.sum(pred_argmax == c)
                diff = pred_count - gt_count
                print(f"    {CLASS_NAMES[c]:12s}: GT={gt_count:4d}  Pred={pred_count:4d}  diff={diff:+d}")
            
            # Confidence analysis
            pred_max = pred.max(axis=-1)
            dynamic_mask = entropy > 0.01
            if np.any(dynamic_mask):
                print(f"\n  Confidence on dynamic cells:")
                print(f"    Mean max prob: {pred_max[dynamic_mask].mean():.3f}")
                print(f"    Max  max prob: {pred_max[dynamic_mask].max():.3f}")
                print(f"    Cells > 0.80:  {np.sum(pred_max[dynamic_mask] > 0.80)}")
                print(f"    Cells > 0.70:  {np.sum(pred_max[dynamic_mask] > 0.70)}")
                print(f"    Cells > 0.60:  {np.sum(pred_max[dynamic_mask] > 0.60)}")
            
            # Top worst cells
            flat_kl = kl.copy()
            flat_kl[~dynamic_mask] = 0
            worst_indices = np.unravel_index(
                np.argsort(flat_kl.ravel())[::-1][:10], kl.shape
            )
            
            print(f"\n  Top 10 worst cells:")
            for i in range(10):
                y, x = worst_indices[0][i], worst_indices[1][i]
                if flat_kl[y, x] < 0.01:
                    break
                init_cls = class_map[y, x]
                gt_cls = np.argmax(gt[y, x])
                pred_cls = np.argmax(pred[y, x])
                print(f"    ({x:2d},{y:2d}) init={CLASS_NAMES[init_cls]:10s} "
                      f"gt={CLASS_NAMES[gt_cls]}({gt[y,x,gt_cls]:.2f}) "
                      f"pred={CLASS_NAMES[pred_cls]}({pred[y,x,pred_cls]:.2f}) "
                      f"KL={kl[y,x]:.3f}")
        
        results[seed_idx] = {
            "score": score,
            "weighted_kl": weighted_kl,
            "n_dynamic": n_dynamic,
        }
        total_score += score
    
    avg_score = total_score / len(gt_data)
    
    print(f"\n{'='*60}")
    print(f"OVERALL: Average score = {avg_score:.1f}")
    print(f"{'='*60}")
    print(f"  (Round 1 actual score was 27.3)")
    print(f"  Improvement: {avg_score - 27.3:+.1f} points")
    
    results["average"] = avg_score
    return results


if __name__ == "__main__":
    import sys
    
    # Find observations file
    obs_path = None
    for candidate in [
        "round_data/observations_*.json",
    ]:
        import glob
        matches = glob.glob(candidate)
        if matches:
            obs_path = matches[0]
            break
    
    # Check for command line args
    mode = sys.argv[1] if len(sys.argv) > 1 else "with_obs"
    
    if mode == "prior_only":
        print("Testing with PRIORS ONLY (no observations)")
        validate_model(observations_path=None, use_knowledge=True)
    elif mode == "no_kb":
        print("Testing WITHOUT knowledge base")
        validate_model(observations_path=obs_path, use_knowledge=False)
    else:
        print("Testing with observations + knowledge base")
        validate_model(observations_path=obs_path, use_knowledge=True)
