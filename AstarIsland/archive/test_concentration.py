"""
Test different Bayesian concentration parameters on R16.
Since we can't score against ground truth, we compare prediction entropy
and distribution shapes across settings.
"""
import numpy as np
import json, os
from client import AstarClient, grid_to_class_map
from config import PROB_FLOOR, NUM_CLASSES
from model_v4 import (
    load_survival_kb, load_expansion_kb,
    estimate_survival_mle, estimate_expansion_from_obs,
    build_prediction_v4,
    DIRICHLET_MIN, DIRICHLET_MAX,
)
import model_v4


def test_concentration(dmin, dmax, obs, detail, survival_kb, expansion_kb, est_surv, est_exp):
    """Build prediction with specific concentration params, return stats."""
    # Temporarily override
    old_min, old_max = model_v4.DIRICHLET_MIN, model_v4.DIRICHLET_MAX
    model_v4.DIRICHLET_MIN = dmin
    model_v4.DIRICHLET_MAX = dmax
    
    all_seeds_settlements = {}
    for si in range(5):
        all_seeds_settlements[si] = detail["initial_states"][si]["settlements"]
    
    all_preds = []
    for seed_idx in range(5):
        state = detail["initial_states"][seed_idx]
        class_map = grid_to_class_map(state["grid"])
        raw_grid = np.array(state["grid"])
        
        pred = build_prediction_v4(
            class_map, raw_grid, state["settlements"],
            obs, seed_idx, 40, 40,
            survival_kb=survival_kb,
            expansion_kb=expansion_kb,
            estimated_survival=est_surv,
            estimated_expansion=est_exp,
            all_seeds_settlements=all_seeds_settlements,
        )
        all_preds.append(pred)
    
    model_v4.DIRICHLET_MIN = old_min
    model_v4.DIRICHLET_MAX = old_max
    return all_preds


def compare_predictions(preds_a, preds_b, label):
    """Compare two prediction sets."""
    total_kl = 0
    total_cells = 0
    max_diff = 0
    high_diff_count = 0
    
    for si in range(len(preds_a)):
        pa, pb = preds_a[si], preds_b[si]
        for y in range(40):
            for x in range(40):
                da, db = pa[y, x], pb[y, x]
                diff = np.max(np.abs(da - db))
                kl = np.sum(da * np.log(np.maximum(da, 1e-10) / np.maximum(db, 1e-10)))
                total_kl += abs(kl)
                total_cells += 1
                max_diff = max(max_diff, diff)
                if diff > 0.02:
                    high_diff_count += 1
    
    print(f"  {label}: avg_kl={total_kl/total_cells:.6f}, max_diff={max_diff:.4f}, "
          f"cells_with_>2%_diff={high_diff_count}/{total_cells}")


def main():
    client = AstarClient()
    budget = client.get_budget()
    round_id = budget["round_id"]
    detail = client.get_round_detail(round_id)
    
    obs_path = f"round_data/observations_{round_id}.json"
    with open(obs_path) as f:
        obs = json.load(f)
    
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    all_seeds_initial_states = {si: detail["initial_states"][si] for si in range(5)}
    est_surv, _ = estimate_survival_mle(obs, all_seeds_initial_states, 40, 40, survival_kb)
    est_exp, _ = estimate_expansion_from_obs(obs, all_seeds_initial_states, 40, 40)
    
    # Test various concentration settings
    configs = [
        ("current (6,40)", 6.0, 40.0),
        ("lower (3,20)", 3.0, 20.0),
        ("lower (4,30)", 4.0, 30.0),
        ("lower (2,15)", 2.0, 15.0),
        ("aggressive (1,10)", 1.0, 10.0),
        ("trust_obs (2,8)", 2.0, 8.0),
    ]
    
    preds_baseline = None
    all_pred_sets = {}
    
    for label, dmin, dmax in configs:
        print(f"\n{label}:")
        preds = test_concentration(dmin, dmax, obs, detail, survival_kb, expansion_kb, est_surv, est_exp)
        all_pred_sets[label] = preds
        
        if preds_baseline is None:
            preds_baseline = preds
        else:
            compare_predictions(preds_baseline, preds, f"vs baseline")
        
        # Compute entropy stats
        total_ent = 0
        for si in range(5):
            for y in range(40):
                for x in range(40):
                    p = preds[si][y, x]
                    ent = -np.sum(p * np.log(np.maximum(p, 1e-10)))
                    total_ent += ent
        avg_ent = total_ent / (5 * 1600)
        print(f"  avg cell entropy: {avg_ent:.4f}")
    
    # Can we estimate which is better? 
    # Idea: for cells observed many times, the empirical distribution is "closer to truth".
    # Compute how well each config matches the empirical distribution for high-obs cells.
    print(f"\n\n=== Empirical distribution match (high-obs cells) ===")
    
    # Build empirical distribution from observations
    obs_counts = np.zeros((5, 40, 40, 6))
    obs_total = np.zeros((5, 40, 40))
    for o in obs:
        si = o["seed_index"]
        vp = o["viewport"]
        for gy in range(vp["h"]):
            for gx in range(vp["w"]):
                mx, my = vp["x"] + gx, vp["y"] + gy
                if 0 <= mx < 40 and 0 <= my < 40:
                    cell_val = o["grid"][gy][gx]
                    from config import TERRAIN_TO_CLASS
                    c = TERRAIN_TO_CLASS.get(cell_val, 0)
                    obs_counts[si, my, mx, c] += 1
                    obs_total[si, my, mx] += 1
    
    # For cells with 5+ observations, compare prediction to empirical
    for label, preds in all_pred_sets.items():
        total_kl = 0
        n_cells = 0
        for si in range(5):
            for y in range(40):
                for x in range(40):
                    n = int(obs_total[si, y, x])
                    if n < 5:
                        continue
                    empirical = obs_counts[si, y, x] / n
                    empirical = np.maximum(empirical, 1e-10)
                    empirical /= empirical.sum()
                    
                    pred = np.maximum(preds[si][y, x], 1e-10)
                    
                    # How well does pred match empirical?
                    kl = np.sum(empirical * np.log(empirical / pred))
                    total_kl += kl
                    n_cells += 1
        
        avg_kl = total_kl / n_cells if n_cells > 0 else 0
        score_est = 100 * np.exp(-3 * avg_kl)
        print(f"  {label:>25s}: avg_kl={avg_kl:.4f}, ~score={score_est:.1f}, n={n_cells}")


if __name__ == "__main__":
    main()
