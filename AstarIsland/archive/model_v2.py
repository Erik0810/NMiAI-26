"""
V2 Prediction Model — GT-Calibrated from Round 1 Analysis.

Key fixes over V1:
1. Uses actual ground truth distributions as priors (not heuristic guesses)
2. Caps confidence — no cell should exceed ~0.65 probability for any class unless static
3. Proper Bayesian updating: GT prior + empirical observations → posterior
4. Smooth spatial variation — predictions change gradually
5. Settlement survival ~40-50%, NOT 80%+ as V1 assumed
6. Ruin probability is tiny (0.01-0.06), NOT 0.20+ as V1 assumed

Ground truth calibration data from Round 1 (5 seeds, ~6400 dynamic cells):

Settlement outcomes:
  - Inland, surrounded by forest:  Emp:0.35 Set:0.44 Rui:0.03 For:0.17
  - Inland, less forest:           Emp:0.38 Set:0.40 Rui:0.03 For:0.18 
  - Coastal:                        Emp:0.32 Set:0.35 Por:0.13 Rui:0.03 For:0.16

Plains (buildable empty land):
  - Near settlement (dist 1):      Emp:0.70 Set:0.24 Rui:0.02 For:0.05
  - Near (dist 2):                  Emp:0.72 Set:0.22 Rui:0.02 For:0.05
  - Medium (dist 3):                Emp:0.73 Set:0.21 Rui:0.02 For:0.05
  - Medium (dist 4):                Emp:0.76 Set:0.18 Rui:0.01 For:0.04
  - Far (dist 5+):                  Emp:0.92 Set:0.07 Rui:0.01 For:0.01

Forest:
  - Near settlement (dist 1):      For:0.64 Set:0.23 Emp:0.10 Rui:0.02 Por:0.02
  - Near (dist 2):                  For:0.65 Set:0.22 Emp:0.10 Rui:0.02 Por:0.01
  - Medium (dist 3):                For:0.68 Set:0.19 Emp:0.10 Rui:0.02 Por:0.02
  - Medium (dist 4):                For:0.71 Set:0.18 Emp:0.08 Rui:0.01 Por:0.02
  - Far (dist 5+):                  For:0.92 Set:0.06 Emp:0.01 Por:0.01

Coastal plains:
  - Near (dist 1-2):               Emp:0.66 Set:0.15 Por:0.15 Rui:0.01 For:0.03
  - Medium (dist 3):                Emp:0.71 Set:0.11 Por:0.14 Rui:0.01 For:0.03
  - Medium (dist 4):                Emp:0.74 Set:0.10 Por:0.11 Rui:0.01 For:0.03
  - Far (dist 5+):                  Emp:0.92 Set:0.04 Por:0.03 For:0.01
"""

import numpy as np
import json
import os
from collections import defaultdict
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES
from learn import KNOWLEDGE_DIR


# ═══════════════════════════════════════════════════════════════════════
# GT-Calibrated Prior Distributions
# ═══════════════════════════════════════════════════════════════════════

# Order: [Empty, Settlement, Port, Ruin, Forest, Mountain]

# Settlement priors by context
SETTLEMENT_PRIORS = {
    # key: (coastal, n_forest_binned)
    # Inland settlements
    ("inland", 0): np.array([0.42, 0.36, 0.01, 0.03, 0.17, 0.01]),
    ("inland", 1): np.array([0.38, 0.40, 0.01, 0.03, 0.17, 0.01]),
    ("inland", 2): np.array([0.37, 0.42, 0.01, 0.03, 0.16, 0.01]),
    ("inland", 3): np.array([0.35, 0.44, 0.01, 0.03, 0.16, 0.01]),
    # Coastal settlements
    ("coastal", 0): np.array([0.32, 0.30, 0.18, 0.03, 0.16, 0.01]),
    ("coastal", 1): np.array([0.32, 0.32, 0.16, 0.03, 0.16, 0.01]),
    ("coastal", 2): np.array([0.32, 0.33, 0.15, 0.03, 0.16, 0.01]),
    ("coastal", 3): np.array([0.32, 0.35, 0.13, 0.03, 0.16, 0.01]),
}

# Plains priors by distance to nearest settlement
PLAINS_INLAND_PRIORS = {
    0: np.array([0.68, 0.25, 0.01, 0.02, 0.03, 0.01]),
    1: np.array([0.70, 0.24, 0.01, 0.02, 0.02, 0.01]),
    2: np.array([0.72, 0.22, 0.01, 0.02, 0.02, 0.01]),
    3: np.array([0.73, 0.21, 0.01, 0.02, 0.02, 0.01]),
    4: np.array([0.76, 0.18, 0.01, 0.01, 0.03, 0.01]),
    5: np.array([0.92, 0.04, 0.01, 0.01, 0.01, 0.01]),  # far
}

PLAINS_COASTAL_PRIORS = {
    0: np.array([0.62, 0.16, 0.16, 0.01, 0.04, 0.01]),
    1: np.array([0.64, 0.15, 0.15, 0.01, 0.04, 0.01]),
    2: np.array([0.67, 0.14, 0.14, 0.01, 0.03, 0.01]),
    3: np.array([0.71, 0.11, 0.14, 0.01, 0.02, 0.01]),
    4: np.array([0.74, 0.10, 0.11, 0.01, 0.03, 0.01]),
    5: np.array([0.92, 0.03, 0.03, 0.01, 0.01, 0.01]),  # far
}

# Forest priors by distance to nearest settlement  
FOREST_PRIORS = {
    0: np.array([0.10, 0.24, 0.02, 0.02, 0.61, 0.01]),
    1: np.array([0.10, 0.23, 0.02, 0.02, 0.62, 0.01]),
    2: np.array([0.10, 0.22, 0.01, 0.02, 0.64, 0.01]),
    3: np.array([0.10, 0.19, 0.02, 0.02, 0.66, 0.01]),
    4: np.array([0.08, 0.18, 0.02, 0.01, 0.70, 0.01]),
    5: np.array([0.01, 0.06, 0.01, 0.01, 0.90, 0.01]),  # far
}

# Static terrain
MOUNTAIN_PRIOR = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
OCEAN_PRIOR = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])

# Maximum confidence for any non-static dynamic cell
MAX_DYNAMIC_CONFIDENCE = 0.70


def load_knowledge_base():
    """Load GT-calibrated knowledge base from learn.py output."""
    kb_path = os.path.join(KNOWLEDGE_DIR, "cumulative_knowledge.json")
    if not os.path.exists(kb_path):
        return None
    with open(kb_path) as f:
        kb = json.load(f)
    
    # Merge all rounds' learnings into single lookup
    merged = {}
    for round_data in kb["rounds"]:
        for key, data in round_data["learnings"].items():
            if key not in merged:
                merged[key] = {"dists": [], "counts": []}
            merged[key]["dists"].append(np.array(data["avg_distribution"]))
            merged[key]["counts"].append(data["count"])
    
    knowledge = {}
    for key, data in merged.items():
        weights = np.array(data["counts"], dtype=float)
        total = weights.sum()
        if total == 0:
            continue
        avg_dist = sum(d * w for d, w in zip(data["dists"], weights)) / total
        knowledge[key] = avg_dist
    
    print(f"  Loaded knowledge base: {len(knowledge)} patterns")
    return knowledge


def _get_cell_prior(cls, raw, x, y, class_map, raw_grid, settlement_set, W, H):
    """
    Get the GT-calibrated prior for a single cell based on its terrain context.
    Returns a 6-element probability array.
    """
    # Static terrain
    if cls == 5:
        return MOUNTAIN_PRIOR.copy()
    if raw == 10:
        return OCEAN_PRIOR.copy()
    
    # Is coastal?
    is_coastal = False
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and raw_grid[ny, nx] == 10:
                is_coastal = True
                break
        if is_coastal:
            break
    
    # Distance to nearest settlement
    min_dist = 999
    for (sx, sy) in settlement_set:
        d = abs(x - sx) + abs(y - sy)
        if d < min_dist:
            min_dist = d
    
    # Count forest neighbors
    n_forest = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and class_map[ny, nx] == 4:
                n_forest += 1
    
    # Initial settlement
    if (x, y) in settlement_set:
        coastal_key = "coastal" if is_coastal else "inland"
        forest_bin = min(n_forest, 3)
        return SETTLEMENT_PRIORS.get(
            (coastal_key, forest_bin),
            SETTLEMENT_PRIORS[("inland", 1)]
        ).copy()
    
    # Forest cell
    if cls == 4:
        dist_bin = min(min_dist, 5)
        return FOREST_PRIORS.get(dist_bin, FOREST_PRIORS[5]).copy()
    
    # Plains/empty cell
    if raw == 11 or cls == 0:
        dist_bin = min(min_dist, 5)
        if is_coastal:
            return PLAINS_COASTAL_PRIORS.get(dist_bin, PLAINS_COASTAL_PRIORS[5]).copy()
        else:
            return PLAINS_INLAND_PRIORS.get(dist_bin, PLAINS_INLAND_PRIORS[5]).copy()
    
    # Ruin (rare initial state)
    if cls == 3:
        return np.array([0.30, 0.15, 0.05, 0.20, 0.28, 0.02])
    
    # Fallback
    return np.array([0.80, 0.05, 0.02, 0.02, 0.10, 0.01])


def _bayesian_update(prior, empirical_counts, total_obs, strength=8.0):
    """
    Bayesian posterior: treat prior as a Dirichlet with pseudo-count 'strength'.
    posterior ∝ prior^strength × likelihood(data).
    
    CRITICAL INSIGHT: With only 1-3 observations per cell, empirical distributions
    are extremely noisy (each obs is ONE stochastic run, GT is average of HUNDREDS).
    We need a VERY STRONG prior to avoid overfitting to individual run outcomes.
    
    With strength=8.0, prior contributes ~80% weight with 2 observations.
    Only with 10+ observations does empirical start to dominate.
    """
    pseudo_counts = prior * strength
    posterior = pseudo_counts + empirical_counts
    posterior = posterior / posterior.sum()
    return posterior


def build_prediction_v2(
    class_map, raw_grid, initial_settlements,
    observations, seed_idx, W, H,
    knowledge=None
):
    """
    Build prediction using GT-calibrated priors + Bayesian updating from observations.
    
    Strategy:
    1. Start with GT-calibrated prior for every cell
    2. For observed cells: Bayesian update prior with empirical distribution
    3. Cap confidence at MAX_DYNAMIC_CONFIDENCE for non-static cells
    4. Apply knowledge base refinements if available
    5. Floor + renormalize
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)
    
    # Settlement positions
    settlement_set = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}
    
    # Build empirical observation counts
    obs_counts = np.zeros((H, W, NUM_CLASSES))
    obs_total = np.zeros((H, W))
    
    # Also track settlement appearances per cell (for survival estimation)
    settlement_appearances = defaultdict(int)
    cell_in_view_count = np.zeros((H, W), dtype=int)
    
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    
    for obs in seed_obs:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        # Track settlement positions in this observation
        obs_sett_positions = {(s["x"], s["y"]) for s in obs["settlements"] 
                             if s.get("alive", True)}
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    cell_val = obs["grid"][gy][gx]
                    cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                    obs_counts[my, mx, cls] += 1
                    obs_total[my, mx] += 1
                    cell_in_view_count[my, mx] += 1
                    
                    if (mx, my) in obs_sett_positions:
                        settlement_appearances[(mx, my)] += 1
    
    # Knowledge base lookup function
    kb_lookup = _build_kb_lookup(knowledge) if knowledge else None
    
    # ── Global parameter estimation ──
    # Estimate how this round's hidden params differ from Round 1 baseline.
    # Round 1 baseline: ~44% settlement survival across 5 seeds.
    global_survival_ratio = 1.0  # multiplier vs Round 1 baseline
    ROUND1_BASELINE_SURVIVAL = 0.44
    
    if seed_obs:
        # Compute observed survival rate for initial settlements
        observed_settlements = 0
        observed_alive = 0
        for (sx, sy), sdata in settlement_set.items():
            if cell_in_view_count[sy, sx] > 0:
                n = cell_in_view_count[sy, sx]
                alive = settlement_appearances.get((sx, sy), 0)
                observed_settlements += n
                observed_alive += alive
        
        if observed_settlements >= 5:
            obs_rate = observed_alive / observed_settlements
            global_survival_ratio = obs_rate / max(0.1, ROUND1_BASELINE_SURVIVAL)
            global_survival_ratio = np.clip(global_survival_ratio, 0.5, 2.0)
            print(f"    Global survival ratio: {global_survival_ratio:.2f} "
                  f"(observed {obs_rate:.2f} vs baseline {ROUND1_BASELINE_SURVIVAL:.2f})")
    
    stats = {"prior": 0, "bayesian": 0, "kb_enhanced": 0}
    
    for y in range(H):
        for x in range(W):
            cls = class_map[y, x]
            raw = raw_grid[y, x]
            
            # Get GT-calibrated prior
            prior = _get_cell_prior(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            
            # Static terrain — just use prior directly
            if cls == 5 or raw == 10:
                pred[y, x] = prior
                continue
            
            # STEP 1: Enhance prior with knowledge base (always, for all cells)
            # KB is from actual GT data — use it almost exclusively when available
            if kb_lookup:
                kb_dist = _kb_prior(
                    cls, raw, x, y, class_map, raw_grid,
                    settlement_set, W, H, kb_lookup
                )
                if kb_dist is not None:
                    prior = 0.95 * kb_dist + 0.05 * prior
                    stats["kb_enhanced"] += 1
            
            # STEP 1.5: Apply global parameter scaling
            # If this round's survival rate differs from Round 1, scale priors
            if global_survival_ratio != 1.0 and not (cls == 5 or raw == 10):
                # Scale settlement/port probability
                old_settle = prior[1] + prior[2]
                if old_settle > 0.02:
                    new_settle = min(0.80, old_settle * global_survival_ratio)
                    delta = new_settle - old_settle
                    # Proportionally adjust settlement and port
                    port_frac = prior[2] / (old_settle + 1e-10)
                    prior[1] += delta * (1 - port_frac)
                    prior[2] += delta * port_frac
                    # Compensate from empty/forest
                    prior[0] -= delta * 0.6
                    prior[4] -= delta * 0.3
                    prior[3] -= delta * 0.05
                    prior = np.maximum(prior, PROB_FLOOR)
                    prior = prior / prior.sum()
            
            # STEP 2: Use observations for settlement survival estimation
            # For all other cells, the KB-enhanced prior is our best prediction.
            # Key insight: with 1-3 obs per cell, the prior (from hundreds of GT 
            # runs) is far more accurate than noisy empirical terrain counts.
            if (x, y) in settlement_set and cell_in_view_count[y, x] >= 2:
                n_views = cell_in_view_count[y, x]
                n_alive = settlement_appearances.get((x, y), 0)
                obs_survival = n_alive / n_views
                prior_survival = prior[1] + prior[2]
                
                # Only adjust if observation strongly disagrees with prior
                if obs_survival < 0.15:
                    # Settlement is probably dead
                    shift = min(0.12, prior_survival * 0.3)
                    prior = prior.copy()
                    prior[1] -= shift * 0.7
                    prior[2] -= shift * 0.1
                    prior[0] += shift * 0.5
                    prior[4] += shift * 0.3
                    prior[3] += shift * 0.05
                    prior = np.maximum(prior, PROB_FLOOR)
                elif obs_survival > 0.85 and n_views >= 3:
                    # Settlement consistently alive
                    shift = min(0.06, (1 - prior_survival) * 0.15)
                    prior = prior.copy()
                    port_frac = prior[2] / (prior[1] + prior[2] + 1e-10)
                    prior[1] += shift * (1 - port_frac)
                    prior[2] += shift * port_frac
                    prior[0] -= shift * 0.6
                    prior[4] -= shift * 0.3
                    prior = np.maximum(prior, PROB_FLOOR)
                
                stats["bayesian"] += 1
            elif obs_total[y, x] >= 8:
                # Many observations: empirical data becomes meaningful
                n = obs_total[y, x]
                strength = max(5.0, 15.0 - n)
                posterior = _bayesian_update(prior, obs_counts[y, x], n, strength)
                prior = posterior
                stats["bayesian"] += 1
            else:
                stats["prior"] += 1
            
            pred[y, x] = prior
    
    # Final floor + renormalize (double pass)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    print(f"    V2 model: {stats['prior']} prior, {stats['bayesian']} bayesian, "
          f"{stats['kb_enhanced']} KB-enhanced, {len(seed_obs)} observations")
    
    return pred


def _refine_settlement_prior(prior, observed_survival, is_coastal, n_views):
    """
    Refine a settlement's prior based on observed survival rate.
    If we see it alive in 8/10 runs, boost settlement probability.
    If we see it alive in 1/10 runs, boost empty/ruin probability.
    """
    refined = prior.copy()
    
    # The observed survival rate tells us how likely this settlement survives
    # Blend between the static prior and the observed rate
    weight = min(0.7, 0.2 * n_views)  # More observations = trust obs more
    
    # Adjust Settlement probability toward observed rate
    base_survival = prior[1] + prior[2]  # settlement + port
    obs_survival = observed_survival
    
    new_survival = (1 - weight) * base_survival + weight * obs_survival
    
    if is_coastal:
        # Split survival between settlement and port
        port_frac = prior[2] / (prior[1] + prior[2] + 1e-10)
        refined[1] = new_survival * (1 - port_frac)
        refined[2] = new_survival * port_frac
    else:
        refined[1] = new_survival
        refined[2] = prior[2]
    
    # Adjust death outcomes proportionally
    death_rate = max(0.01, 1 - new_survival - prior[4] - prior[5] - prior[3])
    refined[0] = death_rate * 0.65  # Empty
    refined[3] = death_rate * 0.10  # Ruin (very low!)
    refined[4] = death_rate * 0.25  # Forest
    
    # Ensure valid distribution
    refined = np.maximum(refined, PROB_FLOOR)
    refined = refined / refined.sum()
    return refined


def _cap_confidence(pred, class_map, raw_grid, settlement_set, W, H):
    """
    Cap the maximum probability for any non-static cell.
    Ground truth rarely exceeds 0.65 for any class on dynamic cells.
    """
    for y in range(H):
        for x in range(W):
            cls = class_map[y, x]
            raw = raw_grid[y, x]
            
            # Skip static terrain (let mountain/ocean stay at 0.95)
            if cls == 5 or raw == 10:
                continue
            
            # Determine how "dynamic" this cell is
            min_dist = min(
                (abs(x - sx) + abs(y - sy) for sx, sy in settlement_set),
                default=999
            )
            
            if min_dist <= 5:
                # Near settlements — highly dynamic
                cap = MAX_DYNAMIC_CONFIDENCE
            elif min_dist <= 8:
                # Moderate distance
                cap = 0.80
            else:
                # Far from settlements — less dynamic, higher confidence OK
                cap = 0.90
            
            # Apply cap
            max_prob = pred[y, x].max()
            if max_prob > cap:
                # Redistribute excess proportionally to other classes
                excess = max_prob - cap
                max_idx = np.argmax(pred[y, x])
                pred[y, x, max_idx] = cap
                
                # Distribute excess to other classes proportionally
                others_sum = pred[y, x].sum() - cap
                if others_sum > 0:
                    for c in range(NUM_CLASSES):
                        if c != max_idx:
                            pred[y, x, c] += excess * (pred[y, x, c] / others_sum)


def _cross_seed_enhance_v2(pred, observations, current_seed, settlement_set,
                            class_map, raw_grid, obs_total, W, H):
    """
    Use observations from other seeds to improve predictions for unobserved cells.
    Since all seeds share the same hidden parameters, expansion patterns are correlated.
    
    Key insight: if other seeds show settlement expansion at (x,y), this seed probably 
    also has expansion there — but modulated by the specific initial settlements.
    """
    other_obs = [o for o in observations if o["seed_index"] != current_seed]
    if not other_obs:
        return
    
    # Collect settlement-presence data from other seeds
    cross_settlement_rate = defaultdict(lambda: {"seen": 0, "has_settlement": 0})
    
    for obs in other_obs:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        obs_sett = {(s["x"], s["y"]) for s in obs["settlements"] if s.get("alive", True)}
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    key = (mx, my)
                    cross_settlement_rate[key]["seen"] += 1
                    if (mx, my) in obs_sett:
                        cross_settlement_rate[key]["has_settlement"] += 1
    
    # For unobserved cells near settlements, use cross-seed data
    transfer_count = 0
    for y in range(H):
        for x in range(W):
            if obs_total[y, x] > 0:
                continue
            if class_map[y, x] == 5 or raw_grid[y, x] == 10:
                continue
            if (x, y) in settlement_set:
                continue
            
            key = (x, y)
            if key in cross_settlement_rate:
                data = cross_settlement_rate[key]
                if data["seen"] >= 2:
                    rate = data["has_settlement"] / data["seen"]
                    if rate > 0.05:
                        # Gentle boost toward settlement based on cross-seed evidence
                        boost = min(0.12, rate * 0.4)
                        pred[y, x, 1] += boost * 0.85  # Settlement
                        pred[y, x, 2] += boost * 0.10  # Port
                        pred[y, x, 3] += boost * 0.05  # Ruin
                        transfer_count += 1
    
    if transfer_count > 0:
        print(f"    Cross-seed: enhanced {transfer_count} unobserved cells")


def _build_kb_lookup(knowledge):
    """Transform knowledge dict for efficient lookup."""
    if not knowledge:
        return None
    return knowledge  # Already a dict of key → distribution array


def _kb_prior(cls, raw, x, y, class_map, raw_grid, settlement_set, W, H, kb_lookup):
    """
    Look up knowledge base for a matching distribution.
    Returns np.array or None if no match found.
    """
    is_coastal = any(
        0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y+dy, x+dx] == 10
        for dy in [-1,0,1] for dx in [-1,0,1] if dx != 0 or dy != 0
    )
    
    is_settlement = (x, y) in settlement_set
    is_port = is_settlement and settlement_set[(x, y)].get("has_port", False)
    
    min_dist = min(
        (abs(x - sx) + abs(y - sy) for sx, sy in settlement_set),
        default=999
    )
    
    n_forest = sum(
        1 for dy in [-1,0,1] for dx in [-1,0,1]
        if (dx != 0 or dy != 0) and 0 <= x+dx < W and 0 <= y+dy < H
        and class_map[y+dy, x+dx] == 4
    )
    
    n_sett_neighbors = sum(
        1 for dy in [-1,0,1] for dx in [-1,0,1]
        if (dx != 0 or dy != 0) and (x+dx, y+dy) in settlement_set
    )
    
    # Build lookup keys in order of specificity
    keys = []
    if is_settlement or is_port:
        stype = "port" if is_port else "settlement"
        coast = "coastal" if is_coastal else "inland"
        keys.append(f"{stype}|{coast}|forest_{min(n_forest,3)}|sett_neighbors_{min(n_sett_neighbors,3)}")
        keys.append(f"{stype}|{coast}|forest_{min(n_forest,3)}")
        keys.append(f"{stype}|{coast}")
    elif cls == 4:
        keys.append(f"forest|near_sett_{min(min_dist,5)}")
    elif cls == 0 and raw == 11:
        coast = "coastal" if is_coastal else "inland"
        keys.append(f"plains|near_sett_{min(min_dist,5)}|{coast}")
        keys.append(f"plains|near_sett_{min(min_dist,5)}")
    
    for key in keys:
        if key in kb_lookup:
            dist = kb_lookup[key]
            if isinstance(dist, np.ndarray):
                return dist.copy()
            return np.array(dist)
    
    return None
