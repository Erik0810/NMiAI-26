"""
V3 Prediction Model — Survival-Rate-Indexed Interpolation.

Key insight from analyzing Rounds 1-5: hidden parameters change DRAMATICALLY
between rounds, causing survival rates to range from 0% (Round 3) to 90% (Round 1).
A fixed KB that averages across rounds is suboptimal.

V3 approach:
1. Estimate the current round's survival rate from observations (all seeds aggregated)
2. Look up per-context-key distributions indexed by survival rate (from Rounds 1-5 GT)
3. Linearly interpolate between the two nearest historical rounds' distributions
4. Adjust individual settlement probabilities based on per-cell observations

This replaces the V2 approach of fixed KB priors + crude global scaling.
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
DEFAULT_SURVIVAL_RATE = 0.28  # Mean of [0.018, 0.235, 0.330, 0.415, 0.419]


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


def _make_context_key(cls, raw, x, y, class_map, raw_grid, settlement_set, W, H):
    """
    Compute the context key for a cell (same scheme as build_kb.py).
    Returns (key_string, is_coastal, min_dist, n_forest) or (None, ...) if static.
    """
    if cls == 5 or raw == 10:
        return None, False, 999, 0
    
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
    
    is_settlement = (x, y) in settlement_set
    is_port = is_settlement and settlement_set[(x, y)].get("has_port", False)
    
    if is_port:
        coast = "coastal" if is_coastal else "inland"
        key = f"port|{coast}|forest_{min(n_forest, 3)}"
    elif is_settlement:
        coast = "coastal" if is_coastal else "inland"
        key = f"settlement|{coast}|forest_{min(n_forest, 3)}"
    elif cls == 4:  # Forest
        dist = min(min_dist, 10)
        if is_coastal:
            key = f"forest|near_sett_{dist}|coastal"
        else:
            key = f"forest|near_sett_{dist}"
    elif raw == 11:  # Plains
        coast = "coastal" if is_coastal else "inland"
        dist = min(min_dist, 10)
        key = f"plains|near_sett_{dist}|{coast}"
    else:
        key = None
    
    return key, is_coastal, min_dist, n_forest


def interpolate_distribution(rates, dists, target_rate):
    """
    Linearly interpolate distribution based on target survival rate.
    rates: sorted list of known survival rates
    dists: corresponding distributions (list of 6-element lists)
    target_rate: the estimated survival rate for current round
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
        # Find bracketing indices
        idx = int(np.searchsorted(rates, target_rate)) - 1
        idx = max(0, min(idx, len(rates) - 2))
        
        r0, r1 = rates[idx], rates[idx + 1]
        if r1 - r0 < 1e-10:
            result = dists[idx].copy()
        else:
            t = (target_rate - r0) / (r1 - r0)
            result = (1 - t) * dists[idx] + t * dists[idx + 1]
    
    # Ensure valid distribution
    result = np.maximum(result, PROB_FLOOR)
    result = result / result.sum()
    return result


def _estimate_survival_rate(observations, all_seeds_settlements, W, H):
    """
    Estimate the current round's GT survival rate from observations.
    
    Uses ALL seeds' observations since hidden parameters are shared across seeds.
    Returns (estimated_survival_rate, confident_bool).
    
    The observed rate from stochastic sims is a direct Monte Carlo estimate of the 
    GT probability. Each observation is an independent sample. We count:
    - How many times initial settlement cells were observed
    - How many times they showed alive (settlement or port)
    
    Also uses expansion observations as secondary signal:
    - If we see many NEW settlements (on non-initial positions), that indicates
      high expansion/survival rates.
    """
    total_observations = 0
    total_alive = 0
    
    # Also track expansion (new settlements on non-initial positions)
    expansion_cells_seen = 0
    expansion_new_settlements = 0
    
    for seed_idx, settlements in all_seeds_settlements.items():
        settlement_set = {(s["x"], s["y"]) for s in settlements if s["alive"]}
        seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
        
        if not seed_obs:
            continue
        
        for obs in seed_obs:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            
            obs_sett = {(s["x"], s["y"]) for s in obs["settlements"]
                       if s.get("alive", True)}
            
            for (sx, sy) in settlement_set:
                if vx <= sx < vx + vw and vy <= sy < vy + vh:
                    total_observations += 1
                    if (sx, sy) in obs_sett:
                        total_alive += 1
            
            # Count expansion: new settlements on non-initial positions
            for gy in range(vh):
                for gx in range(vw):
                    mx, my = vx + gx, vy + gy
                    if 0 <= mx < W and 0 <= my < H:
                        if (mx, my) not in settlement_set:
                            expansion_cells_seen += 1
                            if (mx, my) in obs_sett:
                                expansion_new_settlements += 1
    
    if total_observations < 3:
        return DEFAULT_SURVIVAL_RATE, False
    
    observed_rate = total_alive / total_observations
    
    # Expansion rate as secondary signal
    if expansion_cells_seen > 50:
        expansion_rate = expansion_new_settlements / expansion_cells_seen
        # High expansion rate correlates with high survival
        # Use as a gentle correction
        # Expected expansion rates from historical data:
        #   survival=0.02 (R3): expansion ~0.001
        #   survival=0.23 (R4): expansion ~0.04
        #   survival=0.33 (R5): expansion ~0.06
        #   survival=0.42 (R1/R2): expansion ~0.08
        # Very rough linear: expansion ≈ 0.2 * survival_rate
        expansion_implied_survival = min(0.5, expansion_rate / 0.2)
        
        # Blend: primary is settlement survival, secondary is expansion
        blended = 0.8 * observed_rate + 0.2 * expansion_implied_survival
    else:
        blended = observed_rate
    
    return blended, True


def estimate_survival_mle(observations, all_seeds_initial_states, W, H, survival_kb):
    """
    Maximum Likelihood Estimation of survival rate using ALL observed cell data.
    
    For each candidate survival rate s, compute:
      L(s) = sum over all observed cells: log P(observed_class | context_key, s)
    Return the s that maximizes L(s).
    
    This uses ~10,000+ cell observations (vs ~50 settlement observations in the 
    simple approach), giving much better precision.
    
    Parameters:
        observations: list of observation dicts
        all_seeds_initial_states: dict of seed_idx -> initial state dict with grid + settlements
        W, H: map dimensions
        survival_kb: loaded survival-indexed KB
    """
    context_keys = survival_kb.get("context_keys", {}) if survival_kb else {}
    if not context_keys:
        print("  MLE: No KB context keys, falling back to settlement counting")
        return DEFAULT_SURVIVAL_RATE, False
    
    # Precompute per-seed data
    per_seed_data = {}
    for seed_idx, state in all_seeds_initial_states.items():
        raw_grid = np.array(state["grid"])
        class_map = _grid_to_class_map(state["grid"])
        settlement_set = {(s["x"], s["y"]): s for s in state["settlements"] if s["alive"]}
        per_seed_data[seed_idx] = (raw_grid, class_map, settlement_set)
    
    # Collect samples: (key, observed_class) pairs
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
                    
                    # Skip static terrain (provides no survival info)
                    if cls == 5 or raw == 10:
                        continue
                    
                    # Observed class from simulation
                    cell_val = obs["grid"][gy][gx]
                    obs_class = TERRAIN_TO_CLASS.get(cell_val, 0)
                    
                    # Context key from initial state
                    key, _, _, _ = _make_context_key(
                        cls, raw, mx, my, class_map, raw_grid, settlement_set, W, H
                    )
                    
                    if key and key in context_keys:
                        samples.append((key, obs_class))
                        key_set.add(key)
    
    if len(samples) < 100:
        print(f"  MLE: Only {len(samples)} samples, falling back to settlement counting")
        return DEFAULT_SURVIVAL_RATE, False
    
    # Precompute distributions at candidate rates for all used keys
    candidates = np.linspace(0.005, 0.50, 200)
    key_dists = {}
    for key in key_set:
        entry = context_keys[key]
        dists = np.array([
            interpolate_distribution(entry["rates"], entry["dists"], s)
            for s in candidates
        ])  # shape: (n_candidates, 6)
        key_dists[key] = dists
    
    # Compute log-likelihood for each candidate rate
    ll = np.zeros(len(candidates))
    for key, obs_class in samples:
        probs = key_dists[key][:, obs_class]  # probabilities for this cell at all candidate rates
        ll += np.log(np.maximum(probs, 1e-10))
    
    best_idx = np.argmax(ll)
    best_s = candidates[best_idx]
    
    # Also get simple estimate for comparison
    all_seeds_settlements = {
        si: state["settlements"]
        for si, state in all_seeds_initial_states.items()
    }
    simple_s, _ = _estimate_survival_rate(observations, all_seeds_settlements, W, H)
    
    print(f"  MLE survival estimate: {best_s:.4f} (log-likelihood={ll[best_idx]:.1f}, "
          f"{len(samples)} samples, {len(key_set)} unique keys)")
    print(f"  Simple estimate for reference: {simple_s:.4f}")
    
    return best_s, True


def build_prediction_v3(
    class_map, raw_grid, initial_settlements,
    observations, seed_idx, W, H,
    survival_kb=None,
    estimated_survival=None,
    all_seeds_settlements=None,
):
    """
    Build prediction using survival-rate-indexed interpolation.
    
    Pipeline:
    1. Estimate survival rate from observations (if not provided)
    2. For each cell: compute context key, interpolate distribution from KB
    3. Adjust individual settlements based on per-cell observations
    4. Floor + renormalize
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)
    
    # Settlement positions for this seed
    settlement_set = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}
    
    # Estimate survival rate from ALL seeds' observations
    if estimated_survival is None:
        if all_seeds_settlements is not None:
            estimated_survival, confident = _estimate_survival_rate(
                observations, all_seeds_settlements, W, H
            )
        else:
            # Single-seed fallback
            single_seed = {seed_idx: initial_settlements}
            estimated_survival, confident = _estimate_survival_rate(
                observations, single_seed, W, H
            )
    else:
        confident = True
    
    print(f"    Estimated survival rate: {estimated_survival:.4f}")
    
    # Build per-seed observation data for settlement survival detection
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]
    settlement_appearances = defaultdict(int)
    cell_in_view_count = np.zeros((H, W), dtype=int)
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
                    obs_counts[my, mx, c] += 1
                    obs_total[my, mx] += 1
                    cell_in_view_count[my, mx] += 1
                    
                    if (mx, my) in obs_sett:
                        settlement_appearances[(mx, my)] += 1
    
    # KB context lookup
    context_keys = survival_kb.get("context_keys", {}) if survival_kb else {}
    
    stats = {"interpolated": 0, "fallback": 0, "adjusted": 0}
    
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
            
            # Compute context key
            key, is_coastal, min_dist, n_forest = _make_context_key(
                cls, raw, x, y, class_map, raw_grid, settlement_set, W, H
            )
            
            # Look up survival-indexed distribution
            if key and key in context_keys:
                entry = context_keys[key]
                dist = interpolate_distribution(
                    entry["rates"], entry["dists"], estimated_survival
                )
                stats["interpolated"] += 1
            elif key:
                # Try fallback keys (less specific)
                fallback_dist = _try_fallback_keys(key, context_keys, estimated_survival)
                if fallback_dist is not None:
                    dist = fallback_dist
                    stats["interpolated"] += 1
                else:
                    dist = _hardcoded_fallback(cls, raw, is_coastal, min_dist, n_forest)
                    stats["fallback"] += 1
            else:
                dist = _hardcoded_fallback(cls, raw, is_coastal, min_dist, n_forest)
                stats["fallback"] += 1
            
            # Per-cell settlement survival adjustment (continuous, not binary)
            if (x, y) in settlement_set and cell_in_view_count[y, x] >= 2:
                n_views = cell_in_view_count[y, x]
                n_alive = settlement_appearances.get((x, y), 0)
                obs_survival = n_alive / n_views
                prior_survival = dist[1] + dist[2]
                
                # Compute discrepancy between observation and prior
                delta = obs_survival - prior_survival
                
                # Weight adjustment by confidence (more views = more weight)
                # With 2 views: weight ~0.15, with 5 views: weight ~0.30
                confidence = min(0.35, 0.075 * n_views)
                
                # Only adjust if there's meaningful discrepancy
                if abs(delta) > 0.10:
                    shift = confidence * delta
                    dist = dist.copy()
                    port_frac = dist[2] / (dist[1] + dist[2] + 1e-10)
                    
                    if shift < 0:
                        # Settlement dying: move mass from sett/port to empty/forest
                        dist[1] += shift * (1 - port_frac)
                        dist[2] += shift * port_frac
                        dist[0] -= shift * 0.55  # empty gains
                        dist[4] -= shift * 0.35  # forest gains
                        dist[3] -= shift * 0.05  # ruin gains (tiny)
                    else:
                        # Settlement thriving: move mass from empty/forest to sett/port
                        dist[1] += shift * (1 - port_frac)
                        dist[2] += shift * port_frac
                        dist[0] -= shift * 0.55  # empty loses
                        dist[4] -= shift * 0.35  # forest loses
                    
                    dist = np.maximum(dist, PROB_FLOOR)
                    dist = dist / dist.sum()
                    stats["adjusted"] += 1
            
            pred[y, x] = dist
    
    # Final floor + renormalize (double pass)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    
    print(f"    V3 model: {stats['interpolated']} interpolated, "
          f"{stats['fallback']} fallback, {stats['adjusted']} obs-adjusted, "
          f"{len(seed_obs)} observations")
    
    return pred


def _try_fallback_keys(key, context_keys, survival_rate):
    """Try less specific versions of the key."""
    parts = key.split("|")
    
    # For settlement/port keys with forest count, try different forest counts
    if len(parts) >= 3 and (parts[0] in ("settlement", "port")):
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
                # Try decreasing distances until we find a match
                for d in range(dist - 1, -1, -1):
                    fallback_parts = [pp if not pp.startswith("near_sett_") else f"near_sett_{d}" 
                                     for pp in parts]
                    fallback = "|".join(fallback_parts)
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
    """Hardcoded fallback priors when KB has no match (shouldn't happen often)."""
    dist_bin = min(min_dist, 5)
    
    if cls == 4:  # Forest
        # Interpolate between forest-near and forest-far
        base_forest = 0.60 + 0.06 * dist_bin  # 0.60 to 0.90
        base_sett = max(0.01, 0.24 - 0.036 * dist_bin)
        return np.array([0.10, base_sett, 0.01, 0.02, base_forest, 0.01])
    
    if raw == 11 or cls == 0:  # Plains
        base_empty = 0.68 + 0.05 * dist_bin  # 0.68 to 0.93
        base_sett = max(0.01, 0.24 - 0.04 * dist_bin)
        if is_coastal:
            port = max(0.01, base_sett * 0.5)
            base_sett *= 0.5
            return np.array([base_empty, base_sett, port, 0.01, 0.04, 0.01])
        return np.array([base_empty, base_sett, 0.01, 0.02, 0.04, 0.01])
    
    return np.array([0.80, 0.05, 0.02, 0.02, 0.10, 0.01])


# ═══════════════════════════════════════════════════════════════════════
# Legacy V2 API (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

def load_knowledge_base():
    """Load survival-indexed KB (replaces old cumulative KB loader)."""
    return load_survival_kb()


def build_prediction_v2(*args, **kwargs):
    """Redirect to V3 for backward compatibility."""
    # Translate old knowledge kwarg to new survival_kb
    knowledge = kwargs.pop("knowledge", None)
    kwargs["survival_kb"] = knowledge
    return build_prediction_v3(*args, **kwargs)
