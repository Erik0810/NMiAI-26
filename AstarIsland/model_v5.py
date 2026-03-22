"""
V5 Prediction Model — Settlement Health Signals + Refined Bayesian Blending.

Improvements over V4:
1. SETTLEMENT HEALTH-WEIGHTED OBSERVATIONS: Uses food/population/defense from
   observed settlements to create soft labels. A barely-alive settlement in one
   seed may be dead in others → reduce its settlement probability contribution.
   
2. KB CONFIDENCE-WEIGHTED CONCENTRATION: Bayesian concentration adapts to KB
   data quality. Context keys with more historical data → stronger prior.
   Keys with little data → weaker prior (observations matter more).

3. PORT TRANSITION BOOST: Coastal settlements with high food access (forest
   neighbors) get explicit port probability boost, learned from observation stats.

4. RUIN AWARENESS: Observed dead-settlement positions (ruins/empty where 
   settlements initially were) get stronger observation weights — death signals
   are more diagnostic than survival signals.
"""

import numpy as np
import json
import os
from collections import defaultdict
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from learn import KNOWLEDGE_DIR

# Import V4 components we reuse
from model_v4 import (
    MOUNTAIN_PRIOR, OCEAN_PRIOR,
    DEFAULT_SURVIVAL_RATE, DEFAULT_EXPANSION_RATE,
    DIRICHLET_MIN, DIRICHLET_MAX,
    load_survival_kb, load_expansion_kb,
    _grid_to_class_map,
    _make_context_key, _make_context_key_v3_compat,
    interpolate_distribution, interpolate_distribution_linear,
    interpolate_2d,
    estimate_survival_mle, estimate_expansion_from_obs,
    _try_fallback_keys, _hardcoded_fallback,
)


# ═══════════════════════════════════════════════════════════════════════
# V5-specific parameters
# ═══════════════════════════════════════════════════════════════════════

# Settlement health scoring
HEALTH_FOOD_WEIGHT = 0.5      # Weight of food in health score
HEALTH_POP_WEIGHT = 0.3       # Weight of population (capped at 2.0)
HEALTH_DEFENSE_WEIGHT = 0.2   # Weight of defense

# Soft label parameters: how much to redistribute from settlement class
# based on health. A settlement with health=0 loses SOFT_LABEL_MAX_SHIFT
# probability mass to ruin/empty classes.
SOFT_LABEL_MAX_SHIFT = 0.4    # Max probability mass to shift away
SOFT_LABEL_RUIN_FRAC = 0.6    # Of shifted mass, how much goes to ruin (vs empty)

# KB confidence weighting
KB_COUNT_HIGH = 200            # Total KB data points for "high confidence"
KB_COUNT_LOW = 20              # Total KB data points for "low confidence"
KB_CONF_BOOST = 1.3            # Multiply concentration by this for high-confidence keys
KB_CONF_REDUCE = 0.7           # Multiply concentration by this for low-confidence keys

# Death signal boost: when we observe a settlement died (ruin/empty where
# settlement was), boost the observation weight
DEATH_SIGNAL_BOOST = 1.5       # Weight multiplier for death observations

# Port transition modeling
PORT_BOOST_COASTAL_FOREST = 0.05  # Extra port probability for coastal settlements
                                   # with 2+ forest neighbors and high-survival round


def _compute_settlement_health(settlement_data):
    """
    Compute a 0-1 health score from settlement observation data.
    
    Higher health = more likely to survive across seeds.
    Uses food (0-1), population (0-~4), and defense (0-1).
    """
    food = settlement_data.get("food", 0.5)
    pop = min(settlement_data.get("population", 1.0), 2.0) / 2.0  # normalize to 0-1
    defense = settlement_data.get("defense", 0.5)
    
    health = (
        HEALTH_FOOD_WEIGHT * food +
        HEALTH_POP_WEIGHT * pop +
        HEALTH_DEFENSE_WEIGHT * defense
    )
    return np.clip(health, 0.0, 1.0)


def _compute_kb_confidence(context_keys_entry, target_rate, bandwidth=0.07):
    """
    Compute a confidence score for a KB entry near the target rate.
    Uses the counts field weighted by kernel proximity.
    """
    if context_keys_entry is None:
        return 0.0
    
    # Handle both survival KB ('rates') and expansion KB ('survival_rates')
    if "rates" in context_keys_entry:
        rates = np.array(context_keys_entry["rates"], dtype=float)
    elif "survival_rates" in context_keys_entry:
        rates = np.array(context_keys_entry["survival_rates"], dtype=float)
    else:
        return 0.0
    counts = np.array(context_keys_entry.get("counts", [1] * len(rates)), dtype=float)
    
    # Weighted count using same Gaussian kernel as interpolation
    diff = rates - target_rate
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    
    effective_count = float(np.dot(weights, counts))
    return effective_count


def bayesian_update_v5(prior_dist, obs_counts, n_obs=None, kb_confidence=None):
    """
    V5 Bayesian update with KB confidence-weighted concentration.
    
    Extends V4 by adjusting concentration based on:
    - Number of observations (V4: adaptive 6-40)
    - KB data quality (V5: multiply by confidence factor)
    """
    if n_obs is None:
        n_obs = int(obs_counts.sum())
    
    # Base concentration (same as V4)
    if n_obs <= 1:
        concentration = DIRICHLET_MAX
    elif n_obs >= 5:
        concentration = DIRICHLET_MIN
    else:
        t = (n_obs - 1) / 4.0
        concentration = DIRICHLET_MAX + t * (DIRICHLET_MIN - DIRICHLET_MAX)
    
    # V5: Adjust concentration by KB confidence
    if kb_confidence is not None:
        if kb_confidence >= KB_COUNT_HIGH:
            concentration *= KB_CONF_BOOST
        elif kb_confidence <= KB_COUNT_LOW:
            concentration *= KB_CONF_REDUCE
        else:
            # Linear interpolation
            t = (kb_confidence - KB_COUNT_LOW) / (KB_COUNT_HIGH - KB_COUNT_LOW)
            factor = KB_CONF_REDUCE + t * (KB_CONF_BOOST - KB_CONF_REDUCE)
            concentration *= factor
    
    alpha = prior_dist * concentration
    alpha_post = alpha + obs_counts
    
    result = alpha_post / alpha_post.sum()
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    
    return result


def _build_settlement_health_map(observations, seed_idx, settlement_set):
    """
    Build a map of (x,y) -> average health score from observations.
    Only for cells that are initial settlements.
    """
    health_map = defaultdict(list)  # (x,y) -> list of health scores
    
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    
    for obs in seed_obs:
        obs_sett_map = {(s["x"], s["y"]): s for s in obs["settlements"]
                        if s.get("alive", True)}
        
        # Check initial settlements in viewport
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        for (sx, sy) in settlement_set:
            if vx <= sx < vx + vw and vy <= sy < vy + vh:
                if (sx, sy) in obs_sett_map:
                    # Settlement still alive in this observation
                    health = _compute_settlement_health(obs_sett_map[(sx, sy)])
                    health_map[(sx, sy)].append(health)
                else:
                    # Settlement WAS here initially but died — health = 0
                    health_map[(sx, sy)].append(0.0)
    
    # Average health scores
    avg_health = {}
    for pos, scores in health_map.items():
        avg_health[pos] = float(np.mean(scores))
    
    return avg_health


def _build_soft_obs_counts(observations, seed_idx, settlement_set, W, H):
    """
    Build per-cell observation counts with settlement health-based soft labels.
    
    Instead of hard counts (settlement observed → +1 to settlement class),
    uses soft labels based on settlement health:
    - High health → +1.0 to settlement, as normal
    - Low health → +0.6 to settlement, +0.24 to ruin, +0.16 to empty
    
    Also applies death signal boost for initial settlements observed as dead.
    """
    obs_counts = np.zeros((H, W, NUM_CLASSES))
    obs_total = np.zeros((H, W))
    
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    
    for obs in seed_obs:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        # Map of observed settlements with their stats
        obs_sett_map = {(s["x"], s["y"]): s for s in obs["settlements"]
                        if s.get("alive", True)}
        obs_sett_positions = set(obs_sett_map.keys())
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if not (0 <= mx < W and 0 <= my < H):
                    continue
                
                cell_val = obs["grid"][gy][gx]
                obs_class = TERRAIN_TO_CLASS.get(cell_val, 0)
                
                is_initial_settlement = (mx, my) in settlement_set
                is_observed_alive = (mx, my) in obs_sett_positions
                
                if is_initial_settlement and is_observed_alive:
                    # Settlement still alive — use health-weighted soft label
                    health = _compute_settlement_health(obs_sett_map[(mx, my)])
                    shift = SOFT_LABEL_MAX_SHIFT * (1.0 - health)
                    
                    # Distribute: (1-shift) to observed class, shift to ruin/empty
                    obs_counts[my, mx, obs_class] += (1.0 - shift)
                    obs_counts[my, mx, 3] += shift * SOFT_LABEL_RUIN_FRAC      # ruin
                    obs_counts[my, mx, 0] += shift * (1 - SOFT_LABEL_RUIN_FRAC) # empty
                    obs_total[my, mx] += 1.0
                    
                elif is_initial_settlement and not is_observed_alive:
                    # Settlement DIED — strong death signal
                    obs_counts[my, mx, obs_class] += DEATH_SIGNAL_BOOST
                    obs_total[my, mx] += DEATH_SIGNAL_BOOST
                    
                else:
                    # Non-settlement cell — standard hard count
                    obs_counts[my, mx, obs_class] += 1.0
                    obs_total[my, mx] += 1.0
    
    return obs_counts, obs_total


def _apply_port_boost(dist, is_coastal, n_forest, estimated_survival):
    """
    For coastal settlements (not ports) with high forest access,
    boost the port probability slightly. In high-survival rounds,
    prosperous coastal settlements are more likely to develop ports.
    """
    if not is_coastal:
        return dist
    
    # Only boost when survival is high (prosperous round) and forests nearby
    if estimated_survival < 0.3 or n_forest < 2:
        return dist
    
    # Scale boost by survival rate and forest count
    prosperity = min(1.0, (estimated_survival - 0.3) / 0.3)
    forest_factor = min(1.0, (n_forest - 1) / 2.0)
    boost = PORT_BOOST_COASTAL_FOREST * prosperity * forest_factor
    
    result = dist.copy()
    # Move probability from settlement to port
    transfer = min(boost, result[1] * 0.3)  # Don't transfer more than 30% of settlement prob
    result[2] += transfer   # port class
    result[1] -= transfer   # settlement class
    
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main V5 Prediction Engine
# ═══════════════════════════════════════════════════════════════════════

def build_prediction_v5(
    class_map, raw_grid, initial_settlements,
    observations, seed_idx, W, H,
    survival_kb=None,
    expansion_kb=None,
    estimated_survival=None,
    estimated_expansion=None,
    all_seeds_settlements=None,
    # V5 tunable parameters
    dirichlet_min=None,
    dirichlet_max=None,
    use_health_scoring=True,
    use_port_boost=False,
    use_kb_confidence=False,
    death_boost=1.0,
):
    """
    Build prediction using V5 model.
    
    Pipeline (changes from V4 marked with ★):
    1. Estimate survival + expansion rates from observations (same as V4)
    2. For each cell: compute context key, look up KB distribution (same as V4)
    ★3. Optionally apply port transition boost for coastal + prosperous settlements
    ★4. Build observation counts (optionally with health-weighted soft labels)
    ★5. Bayesian blend with tunable concentration parameters
    6. Floor + renormalize
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)
    
    settlement_set = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}
    
    if estimated_survival is None:
        estimated_survival = DEFAULT_SURVIVAL_RATE
    if estimated_expansion is None:
        estimated_expansion = DEFAULT_EXPANSION_RATE
    
    d_min = dirichlet_min if dirichlet_min is not None else DIRICHLET_MIN
    d_max = dirichlet_max if dirichlet_max is not None else DIRICHLET_MAX
    
    # Build observation counts
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    
    if seed_obs and use_health_scoring:
        obs_counts, obs_total = _build_soft_obs_counts(
            observations, seed_idx, settlement_set, W, H
        )
    elif seed_obs:
        # Standard hard counts (same as V4)
        obs_counts = np.zeros((H, W, NUM_CLASSES))
        obs_total = np.zeros((H, W))
        for obs in seed_obs:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            obs_sett = {(s["x"], s["y"]) for s in obs["settlements"]
                       if s.get("alive", True)}
            for gy in range(vh):
                for gx in range(vw):
                    mx, my = vx + gx, vy + gy
                    if 0 <= mx < W and 0 <= my < H:
                        cell_val = obs["grid"][gy][gx]
                        c = TERRAIN_TO_CLASS.get(cell_val, 0)
                        
                        # Death signal boost
                        is_initial_sett = (mx, my) in settlement_set
                        is_obs_alive = (mx, my) in obs_sett
                        if is_initial_sett and not is_obs_alive and death_boost != 1.0:
                            obs_counts[my, mx, c] += death_boost
                            obs_total[my, mx] += death_boost
                        else:
                            obs_counts[my, mx, c] += 1.0
                            obs_total[my, mx] += 1.0
    else:
        obs_counts = np.zeros((H, W, NUM_CLASSES))
        obs_total = np.zeros((H, W))
    
    # KB context lookup
    context_keys = survival_kb.get("context_keys", {}) if survival_kb else {}
    exp_context_keys = expansion_kb.get("context_keys", {}) if expansion_kb else {}
    
    stats = {"kb_interp": 0, "kb_2d": 0, "fallback": 0, "bayesian_updated": 0}
    
    for y in range(H):
        for x in range(W):
            cls = int(class_map[y, x])
            raw = int(raw_grid[y, x])
            
            # Static terrain
            if cls == 5:
                pred[y, x] = MOUNTAIN_PRIOR.copy()
                continue
            if raw == 10:
                pred[y, x] = OCEAN_PRIOR.copy()
                continue
            
            # ── Step 1: KB-based prior (same as V4) ──
            key_v4, is_coastal, min_dist, n_forest, sett_r5 = _make_context_key(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            key_v3 = _make_context_key_v3_compat(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            
            dist = None
            kb_entry = None
            
            # Try 2D interpolation first
            if key_v3 and key_v3 in exp_context_keys:
                kb_entry = exp_context_keys[key_v3]
                dist = interpolate_2d(
                    kb_entry["survival_rates"], kb_entry["expansion_rates"],
                    kb_entry["dists"], estimated_survival, estimated_expansion
                )
                stats["kb_2d"] += 1
            
            # Fall back to 1D survival interpolation
            if dist is None:
                effective_key = key_v3
                if effective_key and effective_key in context_keys:
                    kb_entry = context_keys[effective_key]
                    dist = interpolate_distribution(
                        kb_entry["rates"], kb_entry["dists"], estimated_survival
                    )
                    stats["kb_interp"] += 1
                elif effective_key:
                    fallback_dist = _try_fallback_keys(effective_key, context_keys, estimated_survival)
                    if fallback_dist is not None:
                        dist = fallback_dist
                        stats["kb_interp"] += 1
                    else:
                        dist = _hardcoded_fallback(cls, raw, is_coastal, min_dist, n_forest)
                        stats["fallback"] += 1
                else:
                    dist = _hardcoded_fallback(cls, raw, is_coastal, min_dist, n_forest)
                    stats["fallback"] += 1
            
            # Optional port transition boost
            if use_port_boost:
                is_settlement = (x, y) in settlement_set
                is_port = is_settlement and settlement_set[(x, y)].get("has_port", False)
                if is_settlement and not is_port and is_coastal:
                    dist = _apply_port_boost(dist, is_coastal, n_forest, estimated_survival)
            
            # Bayesian update with tunable concentration
            n_obs = int(obs_total[y, x])
            if n_obs >= 1:
                cell_counts = obs_counts[y, x]
                
                # Concentration with V5 tunable range
                if n_obs <= 1:
                    concentration = d_max
                elif n_obs >= 5:
                    concentration = d_min
                else:
                    t = (n_obs - 1) / 4.0
                    concentration = d_max + t * (d_min - d_max)
                
                # Optional KB confidence adjustment
                if use_kb_confidence and kb_entry is not None:
                    kb_conf = _compute_kb_confidence(kb_entry, estimated_survival)
                    if kb_conf >= KB_COUNT_HIGH:
                        concentration *= KB_CONF_BOOST
                    elif kb_conf <= KB_COUNT_LOW:
                        concentration *= KB_CONF_REDUCE
                    else:
                        t2 = (kb_conf - KB_COUNT_LOW) / (KB_COUNT_HIGH - KB_COUNT_LOW)
                        concentration *= KB_CONF_REDUCE + t2 * (KB_CONF_BOOST - KB_CONF_REDUCE)
                
                alpha = dist * concentration
                alpha_post = alpha + cell_counts
                dist = alpha_post / alpha_post.sum()
                dist = np.maximum(dist, PROB_FLOOR)
                dist = dist / dist.sum()
                stats["bayesian_updated"] += 1
            
            pred[y, x] = dist
    
    # Final floor + renormalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    print(f"    V5 model: {stats['kb_interp']} 1D, {stats['kb_2d']} 2D, "
          f"{stats['fallback']} fallback, {stats['bayesian_updated']} Bayes, "
          f"d_range=[{d_min},{d_max}], {len(seed_obs)} obs")
    
    return pred
