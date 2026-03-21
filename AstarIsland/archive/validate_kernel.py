"""Quick validation comparing V3 linear vs V4 kernel smoothing oracle scores."""
import numpy as np
import json
import os
import io
import sys
from client import AstarClient, grid_to_class_map
from config import PROB_FLOOR, NUM_CLASSES
import model_v3 as v3_module
from model_v4 import (
    build_prediction_v4, load_survival_kb, load_expansion_kb,
    interpolate_distribution, interpolate_distribution_linear,
)


def compute_score(gt, pred):
    gt_safe = np.maximum(gt, 1e-10)
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)
    entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    dynamic_mask = entropy > 0.01
    if not np.any(dynamic_mask):
        return 100.0
    weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / np.sum(entropy[dynamic_mask])
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


# Suppress all model output
old_stdout = sys.stdout

client = AstarClient()
my_rounds = client.get_my_rounds()
completed = [r for r in my_rounds if r["status"] == "completed"]

sys.stdout = io.StringIO()
survival_kb = load_survival_kb()
expansion_kb = load_expansion_kb()
sys.stdout = old_stdout

round_rates = survival_kb["round_survival_rates"]

print(f"{'Round':>6s}  {'surv':>6s}  {'V3-lin':>7s}  {'V4-kern':>8s}  {'diff':>6s}  {'V4+Bay':>7s}  {'V4+2D':>6s}")
print("-" * 65)

total_v3 = []
total_v4 = []
total_v4b = []

for r in sorted(completed, key=lambda x: x["round_number"]):
    round_id = r["id"]
    rn = r["round_number"]
    detail = client.get_round_detail(round_id)
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    all_seeds_initial_states = {si: detail["initial_states"][si] for si in range(seeds_count)}
    all_seeds_settlements = {si: detail["initial_states"][si]["settlements"] for si in range(seeds_count)}

    # Load observations if available
    obs_path = f"round_data/observations_{round_id}.json"
    observations = json.load(open(obs_path)) if os.path.exists(obs_path) else []

    # Get GT survival rate
    gt_survival = float(round_rates.get(str(rn), 0.28))

    v3_scores = []
    v4_scores = []
    v4b_scores = []
    v4_2d_scores = []

    for si in range(seeds_count):
        try:
            analysis = client.get_analysis(round_id, si)
        except:
            continue
        gt = np.array(analysis["ground_truth"])
        initial_grid = np.array(analysis.get("initial_grid") or detail["initial_states"][si]["grid"])
        class_map = grid_to_class_map(initial_grid.tolist())
        settlements = detail["initial_states"][si]["settlements"]

        sys.stdout = io.StringIO()

        # V3 oracle (linear interp)
        pred_v3 = v3_module.build_prediction_v3(
            class_map, initial_grid, settlements,
            [], si, W, H, survival_kb=survival_kb,
            estimated_survival=gt_survival,
            all_seeds_settlements=all_seeds_settlements,
        )
        v3_scores.append(compute_score(gt, pred_v3))

        # V4 oracle (kernel interp, no Bayesian, no 2D)  
        pred_v4 = build_prediction_v4(
            class_map, initial_grid, settlements,
            [], si, W, H,
            survival_kb=survival_kb, expansion_kb=None,
            estimated_survival=gt_survival,
            all_seeds_settlements=all_seeds_settlements,
        )
        v4_scores.append(compute_score(gt, pred_v4))

        # V4 with Bayesian (kernel + observations)
        if observations:
            pred_v4b = build_prediction_v4(
                class_map, initial_grid, settlements,
                observations, si, W, H,
                survival_kb=survival_kb, expansion_kb=None,
                estimated_survival=gt_survival,
                all_seeds_settlements=all_seeds_settlements,
            )
            v4b_scores.append(compute_score(gt, pred_v4b))

        # V4 with 2D KB (no Bayesian) 
        pred_v4_2d = build_prediction_v4(
            class_map, initial_grid, settlements,
            [], si, W, H,
            survival_kb=survival_kb, expansion_kb=expansion_kb,
            estimated_survival=gt_survival, estimated_expansion=0.13,
            all_seeds_settlements=all_seeds_settlements,
        )
        v4_2d_scores.append(compute_score(gt, pred_v4_2d))

        sys.stdout = old_stdout

    avg_v3 = np.mean(v3_scores) if v3_scores else 0
    avg_v4 = np.mean(v4_scores) if v4_scores else 0
    avg_v4b = np.mean(v4b_scores) if v4b_scores else 0
    avg_v4_2d = np.mean(v4_2d_scores) if v4_2d_scores else 0
    diff = avg_v4 - avg_v3
    obs = "*" if observations else " "

    total_v3.append(avg_v3)
    total_v4.append(avg_v4)
    if avg_v4b:
        total_v4b.append(avg_v4b)

    print(f"R{rn:2d}{obs}   {gt_survival:.3f}  {avg_v3:7.2f}  {avg_v4:8.2f}  {diff:+6.2f}  "
          f"{avg_v4b:7.2f}  {avg_v4_2d:6.2f}")

print("-" * 65)
print(f"{'AVG':>6s}  {'':>6s}  {np.mean(total_v3):7.2f}  {np.mean(total_v4):8.2f}  "
      f"{np.mean(total_v4)-np.mean(total_v3):+6.2f}")
if total_v4b:
    print(f"AVG (obs rounds): V3={np.mean([total_v3[i] for i,r in enumerate(completed) if os.path.exists(f'round_data/observations_{r[\"id\"]}.json')]):.2f}  "
          f"V4+Bay={np.mean(total_v4b):.2f}")
