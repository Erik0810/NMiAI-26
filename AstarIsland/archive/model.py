"""
Advanced prediction model using settlement stats, faction analysis, and expansion modeling.

Key insights from Round 1 observations:
- ~50% of initial settlements survive to year 50
- Massive expansion: 250-370 new settlement positions per seed
- Each sim run is very different → we must build probability distributions
- Settlement stats (population, food, wealth, defense) predict survival
- Faction dominance predicts expansion direction
"""

import numpy as np
from collections import defaultdict
from config import TERRAIN_TO_CLASS, PROB_FLOOR, NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════
# Settlement Stats Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_settlement_stats(observations, initial_states, seed_idx, W, H):
    """
    Analyze settlement stats from observations for one seed.
    Returns per-position statistics and expansion patterns.
    """
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]

    # Track settlement appearances at each position
    position_stats = defaultdict(lambda: {
        "alive_count": 0,
        "total_in_view": 0,
        "populations": [],
        "foods": [],
        "wealths": [],
        "defenses": [],
        "owner_ids": [],
        "is_port": [],
    })

    # Track which cells were in viewport (for computing observation count)
    cell_view_count = np.zeros((H, W), dtype=int)

    for obs in seed_obs:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]

        # Mark all cells in viewport
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    cell_view_count[my, mx] += 1

        # Record settlement stats
        settlement_positions_this_obs = set()
        for s in obs["settlements"]:
            sx, sy = s["x"], s["y"]
            settlement_positions_this_obs.add((sx, sy))
            ps = position_stats[(sx, sy)]
            if s.get("alive", True):
                ps["alive_count"] += 1
                ps["populations"].append(s.get("population", 0))
                ps["foods"].append(s.get("food", 0))
                ps["wealths"].append(s.get("wealth", 0))
                ps["defenses"].append(s.get("defense", 0))
                ps["owner_ids"].append(s.get("owner_id", -1))
                ps["is_port"].append(s.get("has_port", False))

        # Count viewport observations for ALL positions in this viewport
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    position_stats[(mx, my)]["total_in_view"] += 1

    return position_stats, cell_view_count


def compute_survival_predictions(position_stats, initial_settlements, W, H):
    """
    Compute survival probability for each initial settlement based on
    observed stats across multiple simulation runs.
    """
    initial_positions = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}
    survival = {}

    for (x, y), s in initial_positions.items():
        ps = position_stats.get((x, y))
        if ps and ps["total_in_view"] > 0:
            rate = ps["alive_count"] / ps["total_in_view"]
            avg_pop = np.mean(ps["populations"]) if ps["populations"] else 0
            avg_food = np.mean(ps["foods"]) if ps["foods"] else 0
            avg_wealth = np.mean(ps["wealths"]) if ps["wealths"] else 0
            avg_defense = np.mean(ps["defenses"]) if ps["defenses"] else 0
            port_rate = np.mean(ps["is_port"]) if ps["is_port"] else 0

            survival[(x, y)] = {
                "survival_rate": rate,
                "avg_population": avg_pop,
                "avg_food": avg_food,
                "avg_wealth": avg_wealth,
                "avg_defense": avg_defense,
                "port_rate": port_rate,
                "observations": ps["total_in_view"],
                "initial_port": s["has_port"],
            }
        else:
            # Unobserved — use default
            survival[(x, y)] = {
                "survival_rate": 0.50,  # prior from Round 1 data
                "avg_population": 1.0,
                "avg_food": 0.75,
                "avg_wealth": 0.02,
                "avg_defense": 0.40,
                "port_rate": 0.05,
                "observations": 0,
                "initial_port": s["has_port"],
            }

    return survival


# ═══════════════════════════════════════════════════════════════════════
# Expansion Modeling
# ═══════════════════════════════════════════════════════════════════════

def compute_expansion_map(position_stats, initial_settlements, cell_view_count, W, H):
    """
    Compute probability of settlement expansion for each cell.
    A cell has "expansion" if a settlement appeared there in observations
    but it wasn't an initial settlement position.
    """
    initial_positions = {(s["x"], s["y"]) for s in initial_settlements if s["alive"]}

    expansion_prob = np.zeros((H, W))
    port_expansion_prob = np.zeros((H, W))

    for (x, y), ps in position_stats.items():
        if (x, y) in initial_positions:
            continue  # Not expansion, original settlement
        if ps["total_in_view"] == 0:
            continue

        # This is an expansion cell
        settle_rate = ps["alive_count"] / ps["total_in_view"]
        expansion_prob[y, x] = settle_rate

        if ps["is_port"]:
            port_rate = sum(ps["is_port"]) / ps["total_in_view"]
            port_expansion_prob[y, x] = port_rate

    return expansion_prob, port_expansion_prob


# ═══════════════════════════════════════════════════════════════════════
# Faction Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_factions(observations, seed_idx, W, H):
    """
    Analyze faction territory and dominance patterns.
    Returns faction territory maps and conflict zone indicators.
    """
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]

    # Track faction ownership per cell
    cell_factions = defaultdict(lambda: defaultdict(int))

    for obs in seed_obs:
        for s in obs["settlements"]:
            if s.get("alive", True) and "owner_id" in s:
                cell_factions[(s["x"], s["y"])][s["owner_id"]] += 1

    # Compute faction dominance and conflict zones
    faction_dominance = np.zeros((H, W), dtype=int) - 1  # -1 = no faction
    faction_stability = np.zeros((H, W))  # 1.0 = always same owner, 0.0 = contested
    conflict_zone = np.zeros((H, W), dtype=bool)

    for (x, y), factions in cell_factions.items():
        if 0 <= x < W and 0 <= y < H:
            total = sum(factions.values())
            dominant = max(factions, key=factions.get)
            faction_dominance[y, x] = dominant
            faction_stability[y, x] = factions[dominant] / total

            # Conflict zone: multiple factions claim this cell
            if len(factions) > 1:
                conflict_zone[y, x] = True

    # Mark borders between different factions as conflict zones
    for y in range(H):
        for x in range(W):
            if faction_dominance[y, x] < 0:
                continue
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if (faction_dominance[ny, nx] >= 0 and
                                faction_dominance[ny, nx] != faction_dominance[y, x]):
                            conflict_zone[y, x] = True
                            conflict_zone[ny, nx] = True

    return faction_dominance, faction_stability, conflict_zone


# ═══════════════════════════════════════════════════════════════════════
# Stats-Enhanced Prediction Builder
# ═══════════════════════════════════════════════════════════════════════

def build_stats_enhanced_prediction(
    class_map, raw_grid, initial_settlements,
    observations, seed_idx, W, H, knowledge=None
):
    """
    Build prediction using:
    1. Settlement stats (survival rates, port development)
    2. Expansion mapping (where new settlements appeared)
    3. Faction/conflict analysis
    4. Learned knowledge base (if available)
    5. Empirical observations
    """
    pred = np.full((H, W, NUM_CLASSES), PROB_FLOOR)

    # Analyze stats
    position_stats, cell_view_count = analyze_settlement_stats(
        observations, initial_settlements, seed_idx, W, H
    )
    survival = compute_survival_predictions(
        position_stats, initial_settlements, W, H
    )
    expansion_prob, port_expansion_prob = compute_expansion_map(
        position_stats, initial_settlements, cell_view_count, W, H
    )
    faction_dom, faction_stab, conflict_zone = analyze_factions(
        observations, seed_idx, W, H
    )

    # Build empirical observation counts (terrain outcomes)
    obs_counts = np.zeros((H, W, NUM_CLASSES))
    obs_total = np.zeros((H, W))
    seed_obs = [o for o in observations if o["seed_index"] == seed_idx]

    for obs in seed_obs:
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    cell_val = obs["grid"][gy][gx]
                    cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                    obs_counts[my, mx, cls] += 1
                    obs_total[my, mx] += 1

    # Settlement positions for proximity calculations
    settlement_set = {(s["x"], s["y"]): s for s in initial_settlements if s["alive"]}

    stats_used = 0
    expansion_used = 0
    empirical_used = 0
    heuristic_used = 0

    for y in range(H):
        for x in range(W):
            cls = class_map[y, x]
            raw = raw_grid[y, x]

            # ── Static terrain ──
            if cls == 5:  # Mountain
                pred[y, x, 5] = 0.95
                continue
            if raw == 10:  # Ocean
                pred[y, x, 0] = 0.95
                continue

            # ── Cells with direct observations → empirical first ──
            if obs_total[y, x] > 0:
                n = obs_total[y, x]
                empirical = obs_counts[y, x] / n
                alpha = min(0.95, 0.5 + 0.1 * n)

                # For initial settlement positions, enhance with survival stats
                if (x, y) in survival:
                    sv = survival[(x, y)]
                    if sv["observations"] > 0:
                        # Build stats-informed prior
                        stats_prior = _settlement_stats_prior(sv, x, y, raw_grid, W, H)
                        # Blend: empirical dominant, stats as mild correction
                        base = 0.3 * stats_prior + 0.7 * empirical
                        pred[y, x] = alpha * base + (1 - alpha) * pred[y, x]
                        stats_used += 1
                        continue

                pred[y, x] = alpha * empirical + (1 - alpha) * pred[y, x]
                empirical_used += 1
                continue

            # ── Unobserved cells ──

            # Check if this is an initial settlement
            if (x, y) in survival:
                sv = survival[(x, y)]
                pred[y, x] = _settlement_stats_prior(sv, x, y, raw_grid, W, H)
                stats_used += 1
                continue

            # Expansion zones
            if expansion_prob[y, x] > 0:
                ep = expansion_prob[y, x]
                pp = port_expansion_prob[y, x]
                pred[y, x, 1] = ep - pp  # settlement (non-port)
                pred[y, x, 2] = pp       # port
                pred[y, x, 3] = ep * 0.15  # some expand then collapse
                pred[y, x, 0] = max(PROB_FLOOR, 1.0 - ep - ep * 0.15)
                # Forest unlikely at expansion sites
                pred[y, x, 4] = PROB_FLOOR if cls != 4 else 0.05
                expansion_used += 1
                continue

            # ── Heuristic for unobserved, non-settlement cells ──
            min_dist = min(
                (abs(x - sx) + abs(y - sy) for (sx, sy) in settlement_set),
                default=999
            )

            is_coastal = any(
                0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y + dy, x + dx] == 10
                for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if dx != 0 or dy != 0
            )

            in_conflict = conflict_zone[y, x] if 0 <= y < H and 0 <= x < W else False

            if cls == 4:  # Forest
                pred[y, x, 4] = 0.80
                if min_dist <= 2:
                    pred[y, x, 1] = 0.06
                    pred[y, x, 0] = 0.05
                elif min_dist <= 4:
                    pred[y, x, 1] = 0.02
            elif cls == 0 and raw == 11:  # Plains
                pred[y, x, 0] = 0.70
                if min_dist <= 1:
                    # Very close to settlement — high expansion probability
                    pred[y, x, 1] = 0.15
                    pred[y, x, 3] = 0.05
                    if is_coastal:
                        pred[y, x, 2] = 0.06
                elif min_dist <= 3:
                    pred[y, x, 1] = 0.10
                    pred[y, x, 3] = 0.03
                    if is_coastal:
                        pred[y, x, 2] = 0.03
                elif min_dist <= 5:
                    pred[y, x, 1] = 0.05
                    pred[y, x, 3] = 0.01

                # Conflict zones near settlements → more ruins
                if in_conflict and min_dist <= 4:
                    pred[y, x, 3] += 0.04
                    pred[y, x, 1] -= 0.02
            elif cls == 3:  # Ruin (initial)
                pred[y, x, 3] = 0.30
                pred[y, x, 4] = 0.25
                pred[y, x, 0] = 0.20
                pred[y, x, 1] = 0.10
                if is_coastal:
                    pred[y, x, 2] = 0.05
            else:
                pred[y, x, 0] = 0.85

            heuristic_used += 1

    # ── Cross-seed expansion transfer ──
    # Use expansion patterns from other seeds to inform unobserved cells in this seed.
    # If similar terrain context shows expansion on other seeds, boost expectations here.
    _apply_cross_seed_expansion(
        pred, observations, seed_idx, initial_settlements,
        class_map, raw_grid, obs_total, W, H
    )

    # Final: floor + renormalize (double-pass to ensure floor holds after renorm)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    print(f"    Stats-enhanced: {stats_used} stats, {expansion_used} expansion, "
          f"{empirical_used} empirical, {heuristic_used} heuristic")

    return pred


def _apply_cross_seed_expansion(pred, observations, current_seed, initial_settlements,
                                 class_map, raw_grid, obs_total, W, H):
    """
    Transfer expansion patterns from other seeds.
    
    For each unobserved cell that's near settlements and is buildable land:
    - Check if similar cells (same terrain context) on other seeds showed expansion
    - If so, boost settlement probability accordingly
    """
    # Compute average expansion rate from OTHER seeds
    other_seeds = set()
    initial_pos = {(s["x"], s["y"]) for s in initial_settlements if s["alive"]}
    
    cross_expansion_rate = defaultdict(list)  # terrain_key -> list of expansion rates

    for obs in observations:
        if obs["seed_index"] == current_seed:
            continue
        other_seeds.add(obs["seed_index"])
        
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        
        # Count settlements at each position in this observation
        obs_settlements = {(s["x"], s["y"]) for s in obs["settlements"] if s.get("alive", True)}
        
        for gy in range(vh):
            for gx in range(vw):
                mx, my = vx + gx, vy + gy
                if 0 <= mx < W and 0 <= my < H:
                    cell_val = obs["grid"][gy][gx]
                    cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                    # Track if this cell had a settlement in this run
                    had_settlement = (mx, my) in obs_settlements
                    if had_settlement:
                        cross_expansion_rate[(mx, my)].append(1)
                    else:
                        cross_expansion_rate[(mx, my)].append(0)

    if not other_seeds:
        return

    # For unobserved cells in current seed, check cross-seed expansion
    transfer_count = 0
    for y in range(H):
        for x in range(W):
            if obs_total[y, x] > 0:
                continue  # Already has direct observations
            if class_map[y, x] == 5 or raw_grid[y, x] == 10:
                continue  # Static terrain
            if (x, y) in initial_pos:
                continue  # Already handled by survival stats

            # Check cross-seed data for nearby cells with expansion
            # Use a small neighborhood to find relevant cross-seed data
            expansion_signals = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in cross_expansion_rate:
                        rates = cross_expansion_rate[(nx, ny)]
                        avg_rate = np.mean(rates)
                        if avg_rate > 0:
                            # Weight by proximity
                            dist = max(1, abs(dx) + abs(dy))
                            expansion_signals.append(avg_rate / dist)

            if expansion_signals:
                avg_signal = np.mean(expansion_signals)
                if avg_signal > 0.05:  # Meaningful expansion signal
                    boost = min(0.15, avg_signal * 0.3)
                    pred[y, x, 1] += boost  # Settlement
                    pred[y, x, 3] += boost * 0.2  # Some become ruins
                    transfer_count += 1

    if transfer_count > 0:
        print(f"    Cross-seed expansion transfer: {transfer_count} cells boosted "
              f"from {len(other_seeds)} other seeds")


def _settlement_stats_prior(sv, x, y, raw_grid, W, H):
    """
    Build a probability distribution for a settlement cell based on observed stats.
    """
    prior = np.full(NUM_CLASSES, PROB_FLOOR)
    rate = sv["survival_rate"]
    port_rate = sv["port_rate"]

    # Check coastal
    is_coastal = any(
        0 <= x + dx < W and 0 <= y + dy < H and raw_grid[y + dy, x + dx] == 10
        for dy in [-1, 0, 1] for dx in [-1, 0, 1]
        if dx != 0 or dy != 0
    )

    if rate > 0:
        # Settlement survives with some probability
        # Split survival between settlement and port
        prior[1] = rate * (1 - port_rate)  # settlement (non-port)
        prior[2] = rate * port_rate         # port
    
    # Death outcomes
    death_rate = 1 - rate
    if death_rate > 0:
        prior[3] = death_rate * 0.60  # most deaths → ruin
        prior[0] = death_rate * 0.20  # some fade to plains
        prior[4] = death_rate * 0.15  # forest reclaims

    # Coastal boost for port
    if is_coastal and sv["initial_port"]:
        prior[2] += 0.05

    return prior
