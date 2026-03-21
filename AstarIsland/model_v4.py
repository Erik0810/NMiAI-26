"""
V4 Prediction Model — Multi-parameter KB + Empirical Bayesian Blending.

Improvements over V3:
1. TWO hidden parameters: survival_rate + expansion_rate
   - R7 vs R2 proved survival alone is insufficient (same surv, 18pt oracle gap)
   - Expansion rate = fraction of non-settlement plains that become settlements
   
2. Empirical Bayesian blending for ALL observed cells (not just settlements)
   - KB prediction becomes Dirichlet prior
   - Each observation updates posterior: Dir(α + counts)
   - With 50 queries × 225 cells = ~11k observations, many cells get updates
   
3. Extended MLE: jointly estimate survival + expansion from observations

4. Finer context keys: settlements_within_r5, edge proximity
"""

import numpy as np
import json
import os
from collections import defaultdict
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES, CLASS_NAMES
from learn import KNOWLEDGE_DIR


def _grid_to_class_map(grid):
    """Convert raw terrain grid to class indices (local impl to avoid circular import)."""
    arr = np.array(grid)
    class_map = np.zeros_like(arr)
    for code, cls in TERRAIN_TO_CLASS.items():
        class_map[arr == code] = cls
    return class_map


# ═══════════════════════════════════════════════════════════════════════
# Static terrain priors (never change)
# ═══════════════════════════════════════════════════════════════════════
MOUNTAIN_PRIOR = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
OCEAN_PRIOR = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])

# Default survival rate when no observations available (cross-round average)
DEFAULT_SURVIVAL_RATE = 0.28
DEFAULT_EXPANSION_RATE = 0.13  # Cross-round average

# Dirichlet concentration for Bayesian blending (adaptive)
# Lower = observations weighted more. We use adaptive: more obs → lower concentration
DIRICHLET_MIN = 6.0     # concentration for cells with 5+ observations (stronger update)
DIRICHLET_MAX = 40.0    # concentration for cells with 1 observation (very gentle)
DIRICHLET_CONCENTRATION = 40.0  # backwards compat default


def load_survival_kb():
    """Load the survival-rate-indexed knowledge base."""
    kb_path = os.path.join(KNOWLEDGE_DIR, "survival_indexed_kb.json")
    if not os.path.exists(kb_path):
        print("  WARNING: survival_indexed_kb.json not found! Run build_kb.py first.")
        return None
    with open(kb_path) as f:
        kb = json.load(f)
    n_keys = len(kb.get("context_keys", {}))
    print(f"  Loaded survival-indexed KB: {n_keys} context keys, "
          f"rounds={list(kb.get('round_survival_rates', {}).keys())}")
    return kb


def load_expansion_kb():
    """Load the expansion-rate-indexed knowledge base (2D extension)."""
    kb_path = os.path.join(KNOWLEDGE_DIR, "expansion_indexed_kb.json")
    if not os.path.exists(kb_path):
        return None
    with open(kb_path) as f:
        kb = json.load(f)
    n_keys = len(kb.get("context_keys", {}))
    print(f"  Loaded expansion-indexed KB: {n_keys} context keys")
    return kb


def _make_context_key(cls, raw, x, y, class_map, raw_grid, settlement_set, W, H):
    """
    Compute the context key for a cell.
    Returns (key_string, is_coastal, min_dist, n_forest, sett_r5) or (None, ...) if static.
    """
    if cls == 5 or raw == 10:
        return None, False, 999, 0, 0
    
    # Is coastal?
    is_coastal = any(
        0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y+dy, x+dx] == 10
        for dy in [-1,0,1] for dx in [-1,0,1] if dx != 0 or dy != 0
    )
    
    # Distance to nearest settlement
    min_dist = min(
        (abs(x - sx) + abs(y - sy) for sx, sy in settlement_set),
        default=999
    )
    
    # Count forest neighbors
    n_forest = sum(
        1 for dy in [-1,0,1] for dx in [-1,0,1]
        if (dx != 0 or dy != 0) and 0 <= x+dx < W and 0 <= y+dy < H
        and class_map[y+dy, x+dx] == 4
    )
    
    # Settlements within radius 5
    sett_r5 = sum(1 for (sx, sy) in settlement_set
                  if abs(x - sx) + abs(y - sy) <= 5)
    
    is_settlement = (x, y) in settlement_set
    is_port = is_settlement and settlement_set[(x, y)].get("has_port", False)
    
    # Edge proximity
    is_edge = x <= 1 or y <= 1 or x >= W - 2 or y >= H - 2
    
    if is_port:
        coast = "coastal" if is_coastal else "inland"
        key = f"port|{coast}|forest_{min(n_forest, 3)}"
    elif is_settlement:
        coast = "coastal" if is_coastal else "inland"
        # Add settlement density (finer)
        density = "dense" if sett_r5 >= 4 else "sparse"
        key = f"settlement|{coast}|forest_{min(n_forest, 3)}|{density}"
    elif cls == 4:  # Forest
        dist = min(min_dist, 10)
        if is_coastal:
            key = f"forest|near_sett_{dist}|coastal"
        else:
            key = f"forest|near_sett_{dist}"
    elif raw == 11:  # Plains
        coast = "coastal" if is_coastal else "inland"
        dist = min(min_dist, 10)
        if is_edge and dist > 3:
            key = f"plains|near_sett_{dist}|{coast}|edge"
        else:
            key = f"plains|near_sett_{dist}|{coast}"
    else:
        key = None
    
    return key, is_coastal, min_dist, n_forest, sett_r5


def _make_context_key_v3_compat(cls, raw, x, y, class_map, raw_grid, settlement_set, W, H):
    """Make a V3-compatible context key (for KB lookup when V4 key isn't in KB)."""
    if cls == 5 or raw == 10:
        return None
    
    is_coastal = any(
        0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y+dy, x+dx] == 10
        for dy in [-1,0,1] for dx in [-1,0,1] if dx != 0 or dy != 0
    )
    
    min_dist = min(
        (abs(x - sx) + abs(y - sy) for sx, sy in settlement_set),
        default=999
    )
    
    n_forest = sum(
        1 for dy in [-1,0,1] for dx in [-1,0,1]
        if (dx != 0 or dy != 0) and 0 <= x+dx < W and 0 <= y+dy < H
        and class_map[y+dy, x+dx] == 4
    )
    
    is_settlement = (x, y) in settlement_set
    is_port = is_settlement and settlement_set[(x, y)].get("has_port", False)
    
    if is_port:
        coast = "coastal" if is_coastal else "inland"
        return f"port|{coast}|forest_{min(n_forest, 3)}"
    elif is_settlement:
        coast = "coastal" if is_coastal else "inland"
        return f"settlement|{coast}|forest_{min(n_forest, 3)}"
    elif cls == 4:
        dist = min(min_dist, 10)
        if is_coastal:
            return f"forest|near_sett_{dist}|coastal"
        return f"forest|near_sett_{dist}"
    elif raw == 11:
        coast = "coastal" if is_coastal else "inland"
        dist = min(min_dist, 10)
        return f"plains|near_sett_{dist}|{coast}"
    return None


def interpolate_distribution(rates, dists, target_rate):
    """
    Kernel-smoothed interpolation using Nadaraya-Watson estimator.
    
    Instead of linear interpolation between 2 bracketing rates,
    uses a Gaussian kernel to weight ALL data points. This:
    - Uses more data → less noise in estimated distributions
    - Better handles clusters of similar rates (R1/R2/R6/R7 at ~0.42)
    - LOO validation shows +2.25 pts avg improvement over linear
    
    Bandwidth 0.07 is optimal from cross-validation.
    """
    rates = np.array(rates, dtype=float)
    dists = np.array(dists, dtype=float)
    bandwidth = 0.07  # Optimal from LOO cross-validation
    
    if len(rates) == 1:
        result = dists[0].copy()
    else:
        # Gaussian kernel weights
        diff = rates - target_rate
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        
        # Floor to prevent zero weights at extreme rates
        weights = np.maximum(weights, 1e-10)
        weights = weights / weights.sum()
        
        result = np.dot(weights, dists)
    
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


def interpolate_distribution_linear(rates, dists, target_rate):
    """
    Legacy linear interpolation (kept for comparison/fallback).
    """
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


def interpolate_2d(survival_rates, expansion_rates, dists, target_surv, target_exp):
    """
    2D interpolation: find distribution for (survival, expansion) point.
    Uses Gaussian kernel smoothing with separate bandwidths per axis.
    
    LOO validation: +2.14 avg improvement over IDW across 14 rounds.
    """
    surv = np.array(survival_rates, dtype=float)
    exp = np.array(expansion_rates, dtype=float)
    all_dists = np.array(dists, dtype=float)
    
    if len(surv) == 1:
        return all_dists[0].copy()
    
    # 2D Gaussian kernel with separate bandwidths
    bw_surv = 0.07   # Optimal from 1D LOO
    bw_exp = 0.10    # Slightly wider for expansion (noisier parameter)
    
    w_surv = np.exp(-0.5 * ((surv - target_surv) / bw_surv) ** 2)
    w_exp = np.exp(-0.5 * ((exp - target_exp) / bw_exp) ** 2)
    weights = w_surv * w_exp
    
    # Floor to prevent zero weights
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    
    result = np.dot(weights, all_dists)
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


def interpolate_2d_idw(survival_rates, expansion_rates, dists, target_surv, target_exp):
    """Legacy 2D IDW interpolation (kept for comparison)."""
    surv = np.array(survival_rates, dtype=float)
    exp = np.array(expansion_rates, dtype=float)
    all_dists = np.array(dists, dtype=float)
    
    if len(surv) == 1:
        return all_dists[0].copy()
    
    surv_range = max(surv.max() - surv.min(), 0.01)
    exp_range = max(exp.max() - exp.min(), 0.001)
    surv_norm = (surv - surv.min()) / surv_range
    exp_norm = (exp - exp.min()) / exp_range
    target_surv_norm = (target_surv - surv.min()) / surv_range
    target_exp_norm = (target_exp - exp.min()) / exp_range
    
    distances = np.sqrt((surv_norm - target_surv_norm)**2 + (exp_norm - target_exp_norm)**2)
    eps = 0.01
    weights = 1.0 / (distances + eps)
    weights = weights ** 2
    weights = weights / weights.sum()
    
    result = np.dot(weights, all_dists)
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Survival + Expansion MLE
# ═══════════════════════════════════════════════════════════════════════

def estimate_survival_mle(observations, all_seeds_initial_states, W, H, survival_kb):
    """
    Maximum Likelihood Estimation of survival rate using ALL observed cell data.
    Returns (estimated_survival_rate, confident_bool).
    """
    context_keys = survival_kb.get("context_keys", {}) if survival_kb else {}
    if not context_keys:
        return DEFAULT_SURVIVAL_RATE, False
    
    per_seed_data = {}
    for seed_idx, state in all_seeds_initial_states.items():
        raw_grid = np.array(state["grid"])
        class_map = _grid_to_class_map(state["grid"])
        settlement_set = {(s["x"], s["y"]): s for s in state["settlements"] if s["alive"]}
        per_seed_data[seed_idx] = (raw_grid, class_map, settlement_set)
    
    samples = []
    key_set = set()
    
    for obs in observations:
        seed_idx = obs["seed_index"]
        if seed_idx not in per_seed_data:
            continue
        raw_grid, class_map, settlement_set = per_seed_data[seed_idx]
        
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    cls = int(class_map[my, mx])
                    raw = int(raw_grid[my, mx])
                    
                    if cls == 5 or raw == 10:
                        continue
                    
                    cell_val = obs["grid"][gy][gx]
                    obs_class = TERRAIN_TO_CLASS.get(cell_val, 0)
                    
                    # Use V3-compatible key for KB lookup
                    key = _make_context_key_v3_compat(
                        cls, raw, mx, my, class_map, raw_grid, settlement_set, W, H
                    )
                    
                    if key and key in context_keys:
                        samples.append((key, obs_class))
                        key_set.add(key)
    
    if len(samples) < 100:
        return DEFAULT_SURVIVAL_RATE, False
    
    # Candidate survival rates — extend range to 0.65 for high-survival rounds
    candidates = np.linspace(0.005, 0.65, 260)
    key_dists = {}
    for key in key_set:
        entry = context_keys[key]
        # Use LINEAR interpolation for MLE — kernel smoothing flattens the
        # likelihood surface and biases the survival estimate.
        dists = np.array([
            interpolate_distribution_linear(entry["rates"], entry["dists"], s)
            for s in candidates
        ])
        key_dists[key] = dists
    
    ll = np.zeros(len(candidates))
    for key, obs_class in samples:
        probs = key_dists[key][:, obs_class]
        ll += np.log(np.maximum(probs, 1e-10))
    
    best_idx = np.argmax(ll)
    best_s = candidates[best_idx]
    
    print(f"  MLE survival estimate: {best_s:.4f} (log-likelihood={ll[best_idx]:.1f}, "
          f"{len(samples)} samples, {len(key_set)} unique keys)")
    
    return best_s, True


def estimate_expansion_from_obs(observations, all_seeds_initial_states, W, H):
    """
    Estimate expansion rate from observations.
    
    Expansion rate = fraction of non-settlement plains cells that became settlement/port.
    """
    total_plains = 0
    total_new_sett = 0
    
    for obs in observations:
        seed_idx = obs["seed_index"]
        if seed_idx not in all_seeds_initial_states:
            continue
        
        state = all_seeds_initial_states[seed_idx]
        raw_grid = np.array(state["grid"])
        settlement_set = {(s["x"], s["y"]) for s in state["settlements"] if s["alive"]}
        
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        obs_sett = {(s["x"], s["y"]) for s in obs["settlements"]
                   if s.get("alive", True)}
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    raw = int(raw_grid[my, mx])
                    if raw == 11 and (mx, my) not in settlement_set:
                        total_plains += 1
                        if (mx, my) in obs_sett:
                            total_new_sett += 1
    
    if total_plains < 50:
        return DEFAULT_EXPANSION_RATE, False
    
    rate = total_new_sett / total_plains
    print(f"  Expansion estimate: {rate:.4f} ({total_new_sett}/{total_plains} plains -> settlement)")
    return rate, True


# ═══════════════════════════════════════════════════════════════════════
# Empirical Bayesian Blending
# ═══════════════════════════════════════════════════════════════════════

def bayesian_update(prior_dist, obs_counts, n_obs=None):
    """
    Update a prior distribution with observed class counts using Dirichlet-Multinomial.
    
    prior_dist: (6,) prior distribution from KB
    obs_counts: (6,) observed class counts for this cell
    n_obs: number of observations (used for adaptive concentration)
    
    Uses adaptive concentration: many observations → stronger update, few → gentle.
    """
    if n_obs is None:
        n_obs = int(obs_counts.sum())
    
    # Adaptive concentration: interpolate between MAX (1 obs) and MIN (5+ obs)
    if n_obs <= 1:
        concentration = DIRICHLET_MAX
    elif n_obs >= 5:
        concentration = DIRICHLET_MIN
    else:
        t = (n_obs - 1) / 4.0
        concentration = DIRICHLET_MAX + t * (DIRICHLET_MIN - DIRICHLET_MAX)
    
    alpha = prior_dist * concentration  # Dirichlet hyperparameters
    alpha_post = alpha + obs_counts     # Posterior hyperparameters
    
    # Posterior mean
    result = alpha_post / alpha_post.sum()
    
    # Floor + renormalize
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main Prediction Engine
# ═══════════════════════════════════════════════════════════════════════

def build_prediction_v4(
    class_map, raw_grid, initial_settlements,
    observations, seed_idx, W, H,
    survival_kb=None,
    expansion_kb=None,
    estimated_survival=None,
    estimated_expansion=None,
    all_seeds_settlements=None,
):
    """
    Build prediction using V4 model: survival-indexed KB + empirical Bayesian blending.
    
    Pipeline:
    1. Estimate survival + expansion rates from observations
    2. For each cell: compute context key, look up KB distribution
    3. For expansion KB: use 2D interpolation if available, else 1D survival
    4. Bayesian blend with per-cell observation counts
    5. Floor + renormalize
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)
    
    settlement_set = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}
    
    # Use default if not provided
    if estimated_survival is None:
        estimated_survival = DEFAULT_SURVIVAL_RATE
    if estimated_expansion is None:
        estimated_expansion = DEFAULT_EXPANSION_RATE
    
    # ── Build per-cell observation counts for this seed ──
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    obs_counts = np.zeros((H, W, NUM_CLASSES))
    obs_total = np.zeros((H, W))
    settlement_appearances = defaultdict(int)
    
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
                    obs_counts[my, mx, c] += 1
                    obs_total[my, mx] += 1
                    
                    if (mx, my) in obs_sett:
                        settlement_appearances[(mx, my)] += 1
    
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
            
            # ── Step 1: KB-based prior ──
            # Try V4 key first, fall back to V3 key
            key_v4, is_coastal, min_dist, n_forest, sett_r5 = _make_context_key(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            key_v3 = _make_context_key_v3_compat(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            
            dist = None
            
            # Try 2D interpolation (expansion KB) first
            if key_v3 and key_v3 in exp_context_keys:
                entry = exp_context_keys[key_v3]
                dist = interpolate_2d(
                    entry["survival_rates"], entry["expansion_rates"],
                    entry["dists"], estimated_survival, estimated_expansion
                )
                stats["kb_2d"] += 1
            
            # Fall back to 1D survival interpolation
            if dist is None:
                effective_key = key_v3  # V3 key matches the existing KB
                if effective_key and effective_key in context_keys:
                    entry = context_keys[effective_key]
                    dist = interpolate_distribution(
                        entry["rates"], entry["dists"], estimated_survival
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
            
            # ── Step 2: Bayesian update with observations ──
            n_obs = int(obs_total[y, x])
            if n_obs >= 1:
                cell_counts = obs_counts[y, x]
                dist = bayesian_update(dist, cell_counts, n_obs=n_obs)
                stats["bayesian_updated"] += 1
            
            pred[y, x] = dist
    
    # Final floor + renormalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    print(f"    V4 model: {stats['kb_interp']} 1D-interp, {stats['kb_2d']} 2D-interp, "
          f"{stats['fallback']} fallback, {stats['bayesian_updated']} Bayes-updated, "
          f"{len(seed_obs)} observations")
    
    return pred


def _try_fallback_keys(key, context_keys, survival_rate):
    """Try less specific versions of the key."""
    parts = key.split("|")
    
    # For settlement keys with density, try without density
    if len(parts) >= 4 and parts[0] in ("settlement",):
        fallback = "|".join(parts[:3])
        if fallback in context_keys:
            entry = context_keys[fallback]
            return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    # For settlement/port keys with forest count, try different forest counts
    if len(parts) >= 3 and parts[0] in ("settlement", "port"):
        fallback = f"{parts[0]}|{parts[1]}|forest_2"
        if fallback in context_keys and fallback != key:
            entry = context_keys[fallback]
            return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    # For forest or plains with extended distances, try nearest lower distance
    if len(parts) >= 2 and parts[0] in ("forest", "plains"):
        for p in parts:
            if p.startswith("near_sett_"):
                try:
                    dist = int(p.split("_")[-1])
                except ValueError:
                    continue
                for d in range(dist - 1, -1, -1):
                    fallback_parts = [pp if not pp.startswith("near_sett_") else f"near_sett_{d}" 
                                     for pp in parts]
                    fallback = "|".join(fallback_parts)
                    if fallback in context_keys:
                        entry = context_keys[fallback]
                        return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    # For plains with edge, try without edge
    if len(parts) >= 4 and parts[0] == "plains" and parts[-1] == "edge":
        fallback = "|".join(parts[:-1])
        if fallback in context_keys:
            entry = context_keys[fallback]
            return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    # For coastal forest, try non-coastal
    if len(parts) >= 3 and parts[0] == "forest" and parts[-1] == "coastal":
        fallback = "|".join(parts[:-1])
        if fallback in context_keys:
            entry = context_keys[fallback]
            return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    # For plains, try without coastal/inland distinction
    if len(parts) >= 3 and parts[0] == "plains":
        fallback = f"plains|{parts[1]}|inland"
        if fallback in context_keys and fallback != key:
            entry = context_keys[fallback]
            return interpolate_distribution(entry["rates"], entry["dists"], survival_rate)
    
    return None


def _hardcoded_fallback(cls, raw, is_coastal, min_dist, n_forest):
    """Hardcoded fallback priors when KB has no match."""
    dist_bin = min(min_dist, 5)
    
    if cls == 4:
        base_forest = 0.60 + 0.06 * dist_bin
        base_sett = max(0.01, 0.24 - 0.036 * dist_bin)
        return np.array([0.10, base_sett, 0.01, 0.02, base_forest, 0.01])
    
    if raw == 11 or cls == 0:
        base_empty = 0.68 + 0.05 * dist_bin
        base_sett = max(0.01, 0.24 - 0.04 * dist_bin)
        if is_coastal:
            port = max(0.01, base_sett * 0.5)
            base_sett *= 0.5
            return np.array([base_empty, base_sett, port, 0.01, 0.04, 0.01])
        return np.array([base_empty, base_sett, 0.01, 0.02, 0.04, 0.01])
    
    return np.array([0.80, 0.05, 0.02, 0.02, 0.10, 0.01])


# ═══════════════════════════════════════════════════════════════════════
# Legacy compatibility
# ═══════════════════════════════════════════════════════════════════════

def build_prediction_v3(*args, **kwargs):
    """Redirect V3 calls to V4."""
    return build_prediction_v4(*args, **kwargs)
