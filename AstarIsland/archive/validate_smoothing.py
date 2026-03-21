"""
Build smoothed knowledge base and validate parametric vs raw KB interpolation.

Approach: Instead of linear interpolation between 2 bracketing rates,
use Nadaraya-Watson kernel regression with Gaussian kernel across all
14 data points. Also tries adding finer context key splits.
"""

import json
import numpy as np
import os
from collections import defaultdict
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES
from learn import KNOWLEDGE_DIR

def load_gt_data(round_number):
    """Load ground truth analysis for a round (all 5 seeds)."""
    gt_dir = os.path.join("round_data", f"round_{round_number}", "gt_analysis")
    if not os.path.exists(gt_dir):
        return None
    
    seeds = {}
    for seed_idx in range(5):
        seed_path = os.path.join(gt_dir, f"seed_{seed_idx}.json")
        if os.path.exists(seed_path):
            with open(seed_path) as f:
                seeds[seed_idx] = json.load(f)
    return seeds if seeds else None


def load_round_initial_state(round_number):
    """Load the initial grid for a round."""
    # Try round_data/<roundN>/seeds/seed_0_initial.json or similar
    seed_dir = os.path.join("round_data", f"round_{round_number}", "seeds")
    if os.path.exists(seed_dir):
        for fname in os.listdir(seed_dir):
            if "initial" in fname or "seed_0" in fname:
                fpath = os.path.join(seed_dir, fname)
                with open(fpath) as f:
                    return json.load(f)
    
    # Try round detail
    rd_path = os.path.join("round_data", f"round_{round_number}", "round_detail.json")
    if os.path.exists(rd_path):
        with open(rd_path) as f:
            rd = json.load(f)
        if "initial_grid" in rd:
            return {"grid": rd["initial_grid"]}
    
    # Fallback: load from my-rounds API data if available  
    return None


def kernel_interpolation(rates, dists, target_rate, bandwidth=0.06):
    """
    Nadaraya-Watson kernel regression.
    
    Instead of linear interpolation between 2 nearest points,
    use a weighted average of ALL points with Gaussian kernel.
    """
    rates = np.array(rates, dtype=float)
    dists = np.array(dists, dtype=float)
    
    if len(rates) == 1:
        result = dists[0].copy()
    else:
        # Gaussian kernel weights
        diff = rates - target_rate
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        
        # Add small floor to prevent zero weights
        weights = np.maximum(weights, 1e-10)
        weights = weights / weights.sum()
        
        result = np.dot(weights, dists)
    
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


def linear_interpolation(rates, dists, target_rate):
    """Standard linear interpolation (V3/V4 baseline)."""
    rates = np.array(rates, dtype=float)
    dists = np.array(dists, dtype=float)
    
    if len(rates) == 1:
        result = dists[0].copy()
    elif target_rate <= rates[0]:
        result = dists[0].copy()
    elif target_rate >= rates[-1]:
        result = dists[-1].copy()
    else:
        idx = int(np.searchsorted(rates, target_rate)) - 1
        idx = max(0, min(idx, len(rates) - 2))
        r0, r1 = rates[idx], rates[idx + 1]
        if r1 - r0 < 1e-10:
            result = dists[idx].copy()
        else:
            t = (target_rate - r0) / (r1 - r0)
            result = (1 - t) * dists[idx] + t * dists[idx + 1]
    
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


def compute_weighted_kl(gt_dist, pred_dist, epsilon=1e-10):
    """Compute entropy-weighted KL divergence for a single cell."""
    p = np.array(gt_dist, dtype=float)
    q = np.array(pred_dist, dtype=float)
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    
    entropy = -np.sum(p * np.log(p + epsilon))
    kl = np.sum(p * np.log(p / q))
    return entropy * kl, entropy


def leave_one_out_validation(kb, interp_func, **interp_kwargs):
    """
    Leave-one-out cross-validation across rounds.
    For each round R: remove R from KB, interpolate at R's survival rate, compute KL vs R's GT.
    """
    context_keys = kb.get("context_keys", {})
    round_rates = kb.get("round_survival_rates", {})
    
    total_weighted_kl = 0
    total_entropy = 0
    per_round = {}
    
    for leave_out_round, leave_out_rate in round_rates.items():
        round_wkl = 0
        round_ent = 0
        
        for key, entry in context_keys.items():
            rates = entry["rates"]
            dists = entry["dists"]
            
            # Find which index corresponds to this round
            # The rate should match one of the rates in the entry
            # Actually: rates are sorted by survival rate, not by round.
            # We need to know which index corresponds to leave_out_round.
            # This is stored in... let me check.
            pass
        
        per_round[leave_out_round] = {"wkl": round_wkl, "ent": round_ent}
    
    return total_weighted_kl, total_entropy, per_round


def validate_smoothing():
    """
    Main validation: compare linear vs kernel interpolation oracle quality.
    Uses leave-one-out on GT data.
    """
    # Load GT data for all rounds
    kb = json.load(open(os.path.join(KNOWLEDGE_DIR, "survival_indexed_kb.json")))
    round_rates = kb["round_survival_rates"]
    context_keys = kb["context_keys"]
    
    # We need per-round, per-key distributions. The KB has them merged by rate order.
    # To do LOO, we need to identify which index in each key's rates/dists corresponds to each round.
    # The rates are sorted survival rates for all rounds. Since each round contributes one rate,
    # we need to build a mapping: round → rate → index in sorted rates.
    
    # Build round→idx mapping per key
    all_round_numbers = sorted(round_rates.keys(), key=lambda x: int(x))
    n_rounds = len(all_round_numbers)
    
    print(f"KB has {len(context_keys)} keys, {n_rounds} rounds")
    print(f"Round survival rates:")
    for rn in sorted(all_round_numbers, key=lambda x: float(round_rates[x])):
        print(f"  R{rn}: {round_rates[rn]:.4f}")
    
    # For LOO: for each round R, build a reduced KB without R,
    # then interpolate at R's survival rate and compare to R's distribution.
    
    results = {"linear": {}, "kernel": {}}
    bandwidths = [0.03, 0.05, 0.07, 0.10, 0.15]
    for bw in bandwidths:
        results[f"kernel_bw{bw}"] = {}
    
    for leave_out_rn in all_round_numbers:
        leave_rate = float(round_rates[leave_out_rn])
        
        # For each key, do LOO
        for method in list(results.keys()):
            results[method].setdefault(leave_out_rn, {"wkl": 0, "ent": 0, "cells": 0})
        
        for key, entry in context_keys.items():
            rates = np.array(entry["rates"], dtype=float)
            dists = np.array(entry["dists"], dtype=float)
            
            # Find index of this round's rate (closest match)
            idx_match = np.argmin(np.abs(rates - leave_rate))
            gt_dist = dists[idx_match]
            
            # Remove this round
            mask = np.ones(len(rates), dtype=bool)
            mask[idx_match] = False
            
            # Handle duplicate rates (R2=0.415, R6=0.415)
            # If another round has same rate, only remove one
            loo_rates = rates[mask]
            loo_dists = dists[mask]
            
            if len(loo_rates) == 0:
                continue
            
            # Count cells for this key in this round (approx from KB)
            # Use a fixed count since we don't have per-round cell counts in KB
            n_cells = 1  # weight by 1 per key for now
            
            # Compute entropy of GT (for weighting)
            p = np.maximum(gt_dist, 1e-10)
            p = p / p.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            
            if entropy < 0.01:
                continue  # skip near-static cells
            
            # Linear interpolation
            pred_linear = linear_interpolation(loo_rates, loo_dists, leave_rate)
            kl_linear = np.sum(p * np.log(p / np.maximum(pred_linear, 1e-10)))
            results["linear"][leave_out_rn]["wkl"] += entropy * kl_linear
            results["linear"][leave_out_rn]["ent"] += entropy
            results["linear"][leave_out_rn]["cells"] += 1
            
            # Kernel interpolation at various bandwidths
            for bw in bandwidths:
                pred_kernel = kernel_interpolation(loo_rates, loo_dists, leave_rate, bandwidth=bw)
                kl_kernel = np.sum(p * np.log(p / np.maximum(pred_kernel, 1e-10)))
                key_name = f"kernel_bw{bw}"
                results[key_name][leave_out_rn]["wkl"] += entropy * kl_kernel
                results[key_name][leave_out_rn]["ent"] += entropy
                results[key_name][leave_out_rn]["cells"] += 1
    
    # Print results
    print("\n" + "=" * 90)
    print("Leave-One-Out Cross-Validation: Linear vs Kernel Interpolation")
    print("=" * 90)
    
    print(f"\n{'Method':<20s} ", end="")
    for rn in sorted(all_round_numbers, key=lambda x: int(x)):
        print(f" R{rn:>2s}", end="")
    print("   AVG    score")
    print("-" * 90)
    
    for method in ["linear"] + [f"kernel_bw{bw}" for bw in bandwidths]:
        avg_wkl_list = []
        print(f"{method:<20s} ", end="")
        for rn in sorted(all_round_numbers, key=lambda x: int(x)):
            if rn in results[method] and results[method][rn]["ent"] > 0:
                wkl = results[method][rn]["wkl"] / results[method][rn]["ent"]
                score = 100 * np.exp(-3 * wkl) if wkl < 10 else 0
                avg_wkl_list.append(wkl)
                print(f" {score:4.1f}", end="")
            else:
                print(f"   --", end="")
        
        if avg_wkl_list:
            avg_wkl = np.mean(avg_wkl_list)
            avg_score = 100 * np.exp(-3 * avg_wkl)
            print(f"  {avg_score:5.1f}  ({avg_wkl:.4f})")
        else:
            print()
    
    # Print difference from linear
    print(f"\n{'Improvement vs linear':<20s}")
    print("-" * 90)
    
    linear_scores = {}
    for rn in sorted(all_round_numbers, key=lambda x: int(x)):
        if rn in results["linear"] and results["linear"][rn]["ent"] > 0:
            wkl = results["linear"][rn]["wkl"] / results["linear"][rn]["ent"]
            linear_scores[rn] = 100 * np.exp(-3 * wkl)
    
    for method in [f"kernel_bw{bw}" for bw in bandwidths]:
        improvements = []
        print(f"{method:<20s} ", end="")
        for rn in sorted(all_round_numbers, key=lambda x: int(x)):
            if rn in results[method] and results[method][rn]["ent"] > 0:
                wkl = results[method][rn]["wkl"] / results[method][rn]["ent"]
                score = 100 * np.exp(-3 * wkl) if wkl < 10 else 0
                if rn in linear_scores:
                    diff = score - linear_scores[rn]
                    improvements.append(diff)
                    print(f" {diff:+4.1f}", end="")
                else:
                    print(f"   --", end="")
            else:
                print(f"   --", end="")
        
        if improvements:
            print(f"  avg={np.mean(improvements):+.2f}")
        else:
            print()


if __name__ == "__main__":
    validate_smoothing()
