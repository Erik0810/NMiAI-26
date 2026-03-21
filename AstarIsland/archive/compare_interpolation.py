"""
Quick comparison: 2D IDW vs 1D Kernel vs Blended prediction for R16.
Uses the saved observations and current KB to compare approaches.
"""
import numpy as np
import json
import os
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES
from model_v4 import (
    load_survival_kb, load_expansion_kb,
    interpolate_distribution, interpolate_distribution_linear,
    interpolate_2d,
    _make_context_key_v3_compat, _grid_to_class_map,
)


def compare_approaches():
    """Compare KB predictions for various cells using different interpolation."""
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    context_keys = survival_kb.get("context_keys", {})
    exp_context_keys = expansion_kb.get("context_keys", {})
    
    # R16 params
    target_surv = 0.3312
    target_exp = 0.0938
    
    print(f"Target: survival={target_surv:.4f}, expansion={target_exp:.4f}")
    print(f"\nComparing 2D IDW vs 1D Kernel vs 1D Linear for each context key:")
    print(f"{'Key':<40s} {'2D IDW':>8s} {'1D Kern':>8s} {'1D Lin':>8s} {'KL(2D,1K)':>10s}")
    print("-" * 80)
    
    total_kl_2d_vs_kern = 0
    n_keys = 0
    
    for key in sorted(context_keys.keys()):
        entry_1d = context_keys[key]
        
        # 1D kernel
        dist_1d_kern = interpolate_distribution(
            entry_1d["rates"], entry_1d["dists"], target_surv
        )
        
        # 1D linear
        dist_1d_lin = interpolate_distribution_linear(
            entry_1d["rates"], entry_1d["dists"], target_surv
        )
        
        # 2D IDW
        if key in exp_context_keys:
            entry_2d = exp_context_keys[key]
            dist_2d = interpolate_2d(
                entry_2d["survival_rates"], entry_2d["expansion_rates"],
                entry_2d["dists"], target_surv, target_exp
            )
        else:
            dist_2d = dist_1d_kern
        
        # Compute symmetric KL divergence between 2D and 1D kernel
        kl = np.sum(dist_2d * np.log(dist_2d / np.maximum(dist_1d_kern, 1e-10)))
        
        # Show the max probability class for each
        c2d = np.argmax(dist_2d)
        c1k = np.argmax(dist_1d_kern)
        c1l = np.argmax(dist_1d_lin)
        
        marker = " <<<" if kl > 0.05 else ""
        print(f"{key:<40s} {dist_2d[c2d]:.4f}[{c2d}] {dist_1d_kern[c1k]:.4f}[{c1k}] "
              f"{dist_1d_lin[c1l]:.4f}[{c1l}] {kl:10.6f}{marker}")
        
        total_kl_2d_vs_kern += kl
        n_keys += 1
    
    print(f"\nAvg KL(2D, 1D-kernel) = {total_kl_2d_vs_kern / n_keys:.6f}")
    print(f"Total keys compared: {n_keys}")
    
    # Also look at a few important keys in detail
    print(f"\n\n{'='*80}")
    print("Detailed comparison for high-error keys (plains|near_sett_1-3|inland)")
    print(f"{'='*80}")
    
    detail_keys = [
        "plains|near_sett_1|inland",
        "plains|near_sett_2|inland",
        "plains|near_sett_3|inland",
        "settlement|near_sett_0|forest_2",
        "forest|near_sett_1|coastal",
        "port|near_sett_0|forest_2",
    ]
    
    for key in detail_keys:
        if key not in context_keys:
            continue
        entry_1d = context_keys[key]
        
        dist_kern = interpolate_distribution(entry_1d["rates"], entry_1d["dists"], target_surv)
        dist_lin = interpolate_distribution_linear(entry_1d["rates"], entry_1d["dists"], target_surv)
        
        if key in exp_context_keys:
            entry_2d = exp_context_keys[key]
            dist_2d = interpolate_2d(
                entry_2d["survival_rates"], entry_2d["expansion_rates"],
                entry_2d["dists"], target_surv, target_exp
            )
        else:
            dist_2d = dist_kern
        
        # Blend: 60% 1D kernel + 40% 2D
        blend = 0.6 * dist_kern + 0.4 * dist_2d
        blend = np.maximum(blend, PROB_FLOOR)
        blend /= blend.sum()
        
        classes = ["Empty", "Settl", "Port", "Ruin", "Forst", "Mount"]
        print(f"\n  {key}:")
        print(f"    {'':>10s} " + " ".join(f"{c:>7s}" for c in classes))
        print(f"    {'2D IDW':>10s} " + " ".join(f"{v:7.4f}" for v in dist_2d))
        print(f"    {'1D Kernel':>10s} " + " ".join(f"{v:7.4f}" for v in dist_kern))
        print(f"    {'1D Linear':>10s} " + " ".join(f"{v:7.4f}" for v in dist_lin))
        print(f"    {'Blend 6:4':>10s} " + " ".join(f"{v:7.4f}" for v in blend))
        
        # Show which round data points are closest
        rates = np.array(entry_1d["rates"])
        diffs = np.abs(rates - target_surv)
        closest_idx = np.argsort(diffs)[:3]
        print(f"    Closest 1D rates: {[f'{rates[i]:.3f}' for i in closest_idx]}")
        
        if key in exp_context_keys:
            entry_2d = exp_context_keys[key]
            surv = np.array(entry_2d["survival_rates"])
            exp_r = np.array(entry_2d["expansion_rates"])
            dist2d = np.sqrt((surv - target_surv)**2 + (exp_r - target_exp)**2)
            closest_2d = np.argsort(dist2d)[:3]
            print(f"    Closest 2D points: {[(f's={surv[i]:.3f}', f'e={exp_r[i]:.3f}') for i in closest_2d]}")


if __name__ == "__main__":
    compare_approaches()
