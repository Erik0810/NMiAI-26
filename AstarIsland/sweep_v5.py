"""
V5 Parameter Sweep — find optimal Bayesian concentration + feature flags.

Tests multiple V5 configurations against V4 baseline on rounds with observations.
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


def sweep():
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r["status"] == "completed"]
    
    survival_kb = load_survival_kb()
    expansion_kb = load_expansion_kb()
    
    # Only test on rounds WITH observations (where Bayesian update matters)
    obs_rounds = []
    for r in sorted(completed, key=lambda x: x["round_number"]):
        round_id = r["id"]
        obs_path = f"round_data/observations_{round_id}.json"
        if os.path.exists(obs_path):
            obs_rounds.append(r)
    
    print(f"Testing on {len(obs_rounds)} rounds with observations")
    
    # Pre-load all round data
    round_cache = {}
    for r in obs_rounds:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W, H = detail["map_width"], detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        with open(f"round_data/observations_{round_id}.json") as f:
            observations = json.load(f)
        
        all_seeds_initial_states = {si: detail["initial_states"][si] for si in range(seeds_count)}
        all_seeds_settlements = {si: detail["initial_states"][si]["settlements"] for si in range(seeds_count)}
        
        # Ground truth rates
        gt_survivals, gt_expansions = [], []
        seed_gts = {}
        for si in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, si)
                gt_data = np.array(analysis["ground_truth"])
                seed_gts[si] = (gt_data, analysis)
                settlements = all_seeds_settlements[si]
                settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
                for (sx, sy) in settlement_set:
                    gt_survivals.append(float(gt_data[sy, sx, 1] + gt_data[sy, sx, 2]))
                initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][si]["grid"])
                for yc in range(H):
                    for xc in range(W):
                        if int(initial_grid[yc, xc]) == 11 and (xc, yc) not in settlement_set:
                            gt_expansions.append(float(gt_data[yc, xc, 1] + gt_data[yc, xc, 2]))
            except:
                pass
        
        gt_survival = np.mean(gt_survivals) if gt_survivals else 0.28
        gt_expansion = np.mean(gt_expansions) if gt_expansions else 0.13
        
        # MLE estimates
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        est_survival, _ = estimate_survival_mle(observations, all_seeds_initial_states, W, H, survival_kb)
        est_expansion, _ = estimate_expansion_from_obs(observations, all_seeds_initial_states, W, H)
        sys.stdout = old_stdout
        
        round_cache[rn] = {
            "detail": detail, "observations": observations,
            "all_seeds_initial_states": all_seeds_initial_states,
            "all_seeds_settlements": all_seeds_settlements,
            "seed_gts": seed_gts,
            "gt_survival": gt_survival, "gt_expansion": gt_expansion,
            "est_survival": est_survival, "est_expansion": est_expansion,
            "W": W, "H": H, "seeds_count": seeds_count,
        }
        print(f"  Cached R{rn}: surv={gt_survival:.3f}, exp={gt_expansion:.3f}")
    
    # ── Define configurations to test ──
    configs = [
        # (name, kwargs for build_prediction_v5)
        ("V4 baseline",         {"dirichlet_min": 6.0, "dirichlet_max": 40.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Lower conc [4,25]",   {"dirichlet_min": 4.0, "dirichlet_max": 25.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Lower conc [3,20]",   {"dirichlet_min": 3.0, "dirichlet_max": 20.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Lower conc [2,15]",   {"dirichlet_min": 2.0, "dirichlet_max": 15.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Very low [1.5,10]",   {"dirichlet_min": 1.5, "dirichlet_max": 10.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Ultra low [1,5]",     {"dirichlet_min": 1.0, "dirichlet_max": 5.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Higher conc [8,50]",  {"dirichlet_min": 8.0, "dirichlet_max": 50.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.0}),
        ("Death boost 1.5",     {"dirichlet_min": 6.0, "dirichlet_max": 40.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.5}),
        ("Death boost 2.0",     {"dirichlet_min": 6.0, "dirichlet_max": 40.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 2.0}),
        ("Low+death [3,20]+1.5",{"dirichlet_min": 3.0, "dirichlet_max": 20.0, "use_health_scoring": False, "use_port_boost": False, "death_boost": 1.5}),
        ("Health scoring",      {"dirichlet_min": 6.0, "dirichlet_max": 40.0, "use_health_scoring": True, "use_port_boost": False, "death_boost": 1.0}),
        ("Health+low [3,20]",   {"dirichlet_min": 3.0, "dirichlet_max": 20.0, "use_health_scoring": True, "use_port_boost": False, "death_boost": 1.0}),
        ("Port boost",          {"dirichlet_min": 6.0, "dirichlet_max": 40.0, "use_health_scoring": False, "use_port_boost": True, "death_boost": 1.0}),
    ]
    
    print(f"\n{'='*100}")
    print(f"PARAMETER SWEEP — {len(configs)} configurations × {len(obs_rounds)} rounds")
    print(f"{'='*100}")
    
    # Header
    round_nums = sorted(round_cache.keys())
    header = f"{'Config':25s} |"
    for rn in round_nums:
        header += f" R{rn:2d} "
    header += f"| {'Avg':>6s} {'Δ':>6s}"
    print(header)
    print("-" * len(header))
    
    # First run V4 to get baseline
    v4_scores = {}
    for rn, cache in round_cache.items():
        scores = []
        for si, (gt_data, analysis) in cache["seed_gts"].items():
            initial_grid = np.array(analysis.get("initial_grid") or cache["detail"]["initial_states"][si]["grid"])
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = cache["detail"]["initial_states"][si]["settlements"]
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            pred = build_prediction_v4(
                class_map, initial_grid, settlements,
                cache["observations"], si, cache["W"], cache["H"],
                survival_kb=survival_kb, expansion_kb=expansion_kb,
                estimated_survival=cache["est_survival"],
                estimated_expansion=cache["est_expansion"],
                all_seeds_settlements=cache["all_seeds_settlements"],
            )
            sys.stdout = old_stdout
            sc, _, _ = compute_score(gt_data, pred)
            scores.append(sc)
        v4_scores[rn] = np.mean(scores)
    
    v4_avg = np.mean(list(v4_scores.values()))
    line = f"{'V4 actual (reference)':25s} |"
    for rn in round_nums:
        line += f" {v4_scores[rn]:4.1f}"
    line += f"| {v4_avg:6.2f} {'---':>6s}"
    print(line)
    
    # Run each V5 config
    best_config = None
    best_avg = v4_avg
    
    for config_name, config_kwargs in configs:
        config_scores = {}
        for rn, cache in round_cache.items():
            scores = []
            for si, (gt_data, analysis) in cache["seed_gts"].items():
                initial_grid = np.array(analysis.get("initial_grid") or cache["detail"]["initial_states"][si]["grid"])
                class_map = grid_to_class_map(initial_grid.tolist())
                settlements = cache["detail"]["initial_states"][si]["settlements"]
                
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    pred = build_prediction_v5(
                        class_map, initial_grid, settlements,
                        cache["observations"], si, cache["W"], cache["H"],
                        survival_kb=survival_kb, expansion_kb=expansion_kb,
                        estimated_survival=cache["est_survival"],
                        estimated_expansion=cache["est_expansion"],
                        all_seeds_settlements=cache["all_seeds_settlements"],
                        **config_kwargs,
                    )
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"  ERR {config_name} R{rn} s{si}: {e}")
                    continue
                finally:
                    sys.stdout = old_stdout
                
                sc, _, _ = compute_score(gt_data, pred)
                scores.append(sc)
            
            config_scores[rn] = np.mean(scores) if scores else 0
        
        avg = np.mean(list(config_scores.values()))
        delta = avg - v4_avg
        
        line = f"{config_name:25s} |"
        for rn in round_nums:
            d = config_scores[rn] - v4_scores[rn]
            line += f" {config_scores[rn]:4.1f}"
        line += f"| {avg:6.2f} {delta:+5.2f}"
        if delta > 0:
            line += " ✓"
        print(line)
        
        if avg > best_avg:
            best_avg = avg
            best_config = config_name
    
    print(f"\n{'='*100}")
    if best_config:
        print(f"BEST: {best_config} (avg={best_avg:.2f}, Δ={best_avg - v4_avg:+.2f})")
    else:
        print(f"V4 baseline is already optimal (avg={v4_avg:.2f})")
    
    # Also show oracle comparison
    print(f"\nOracle parameter comparison:")
    for rn, cache in round_cache.items():
        scores_orc = []
        for si, (gt_data, analysis) in cache["seed_gts"].items():
            initial_grid = np.array(analysis.get("initial_grid") or cache["detail"]["initial_states"][si]["grid"])
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = cache["detail"]["initial_states"][si]["settlements"]
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            pred = build_prediction_v4(
                class_map, initial_grid, settlements,
                cache["observations"], si, cache["W"], cache["H"],
                survival_kb=survival_kb, expansion_kb=expansion_kb,
                estimated_survival=cache["gt_survival"],
                estimated_expansion=cache["gt_expansion"],
                all_seeds_settlements=cache["all_seeds_settlements"],
            )
            sys.stdout = old_stdout
            sc, _, _ = compute_score(gt_data, pred)
            scores_orc.append(sc)
        orc = np.mean(scores_orc)
        mle = v4_scores[rn]
        print(f"  R{rn:2d}: oracle={orc:.1f}, MLE={mle:.1f}, gap={orc-mle:.2f}")


if __name__ == "__main__":
    sweep()
