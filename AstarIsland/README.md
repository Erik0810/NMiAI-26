# Astar Island Solver

Predict a stochastic Norse civilisation simulator's final state from limited viewport observations.

**Platform**: [app.ainm.no](https://app.ainm.no) | **API**: api.ainm.no/astar-island/  
**Team**: Pining for the fjords

---

## Current Status (2026-03-22)

| Round | Score | Weight | Weighted | Model | Survival | Expansion |
|-------|------:|-------:|---------:|-------|:--------:|:---------:|
| R1  | 27.3 | 1.050 | 28.7  | V1 (heuristic)     | 0.419 | 0.170 |
| R2  | 80.1 | 1.103 | 88.3  | V2 (GT-calibrated) | 0.415 | 0.205 |
| R6  | 78.2 | 1.340 | 104.8 | V3 (6-round KB)    | 0.415 | 0.264 |
| R7  | 64.2 | 1.407 | 90.4  | V3 (6-round KB)    | 0.423 | 0.147 |
| R10 | 86.6 | 1.629 | 141.1 | V3 (9-round KB)    | 0.058 | 0.009 |
| R14 | 77.4 | 1.980 | 153.3 | V3 (13-round KB)   | 0.522 | 0.265 |
| R15 | 89.0 | 2.079 | 184.9 | V4 (14-round KB)   | 0.328 | 0.187 |
| R16 | 81.0 | 2.183 | 176.8 | V4 (15-round KB)   | 0.294 | 0.080 |
| R17 | 85.2 | 2.292 | 195.4 | V4 (16-round KB)   | 0.454 | 0.290 |
| R18 | 74.3 | 2.407 | 178.7 | V4 (17-round KB)   | 0.632 | 0.381 |
| R19 | 87.0 | 2.527 | **219.9** | V4 (18-round KB) | 0.041 | 0.012 |
| R22 | --   | 2.925 | --    | V4 (19-round KB)   | est 0.130 | est 0.030 |

**Best weighted**: R19 = 87.0 x 2.527 = **219.9**  
**Leaderboard**: #1 is 261.2. Gap is about 41 points.

---

## Score history and plateaus

### Plateau 1: The 80-point wall (R2 through R7)

V1 was a disaster at 27.3. V2 jumped to 80 by calibrating from ground truth, but then we got stuck. R6 scored 78.2, R7 dropped to 64.2. The model was averaging distributions across rounds that had wildly different hidden parameters. A round with 2% survival and a round with 42% survival were getting blended into the same lookup table. The fix was indexing the KB by survival rate so the model could pick the right regime.

### Plateau 2: The 87-point ceiling (R10 through R14)

R10 hit 86.6 which felt great at the time. Then R14 came in at 77.4 despite having more training data. The problem was twofold: linear interpolation between only two KB neighbours was noisy, and we were missing a second hidden parameter (expansion rate) that controls how fast settlements spread to new plains cells. Plains near settlements accounted for 39% of total prediction error. Adding the expansion dimension to the KB and switching to Gaussian kernel smoothing across all historical rounds broke through to 89.0 on R15.

### Plateau 3: The MLE extrapolation failure (R18)

R18 scored 74.3 which was our worst result since R7. The cause: R18 had the highest survival rate we had ever seen (0.632) and also the highest expansion (0.381). The MLE search range was capped at 0.65 and linear interpolation clamped at the edge of the KB (max 0.599), so the model literally could not represent what was happening. The MLE found 0.508 when the true value was 0.632. Fixes: extended MLE range to [0.005, 0.85], added linear extrapolation beyond KB edges instead of clamping, and rebuilt the KB with R18 data. Retroactive testing showed these fixes would have scored 85.0 instead of 74.3.

### Current situation

R19 bounced back to 87.0 (low-survival round, well within KB range). The best weighted is now 219.9 but the top of the leaderboard is at 261. That gap is roughly one round of high performance on a recent (high-weight) round. The model is well calibrated for survival rates 0.02-0.63, but extreme values beyond the KB range remain a risk. Each new round helps because it adds another data point to interpolate from.

---

## What moved the score (27 to 89)

### 1. Survival-Indexed Knowledge Base (V2 to V3, +50 pts)

V1/V2 used static lookup tables. The problem is that hidden parameters shift every round: settlement survival ranges from 2% (R3) to 60% (R12), so a fixed prior is wrong for almost every round.

The fix was to store all ground-truth distributions indexed by each round's survival rate. At prediction time, estimate the current survival rate and interpolate from the KB. V1 scored 27, V2 scored 80, V3 hit **86.6** on R10.

### 2. MLE Survival Estimation (V3, +5-10 pts per round)

Counting surviving settlements from viewport observations gives you maybe 50 data points, and if the viewports miss the right areas the estimate is way off.

Instead: MLE over **all ~10,000 observed cells**. Every cell type (plains, forest, port, ruin) carries information about the hidden survival rate via the likelihood model P(observed_class | context_key, survival_rate).

| Method | R1 err | R2 err | R10 err |
|--------|-------:|-------:|--------:|
| Count-based | 0.054 | 0.026 | 0.001 |
| **Full-cell MLE** | **0.001** | **0.007** | **0.028** |

### 3. Expansion Parameter Discovery (+2-5 pts)

Even with accurate survival estimates, near-settlement plains cells had persistent errors. KB error analysis showed `plains|near_sett_1-3|inland` alone accounted for **39.4%** of total weighted KL error.

Dug into it with `find_hidden_params.py` and found a second hidden parameter: the **expansion rate**, the probability that plains cells become new settlements. Ranges from 0.002 (R3) to 0.292 (R11). Built a 2D KB indexed by both parameters.

### 4. 2D Gaussian Kernel Smoothing (V4, +2.1 pts avg)

Linear interpolation only uses the two nearest KB neighbours and throws away everything else.

Replaced with Nadaraya-Watson kernel regression (Gaussian kernel, bw_surv=0.07, bw_exp=0.10). Every historical round contributes, weighted by proximity in the survival/expansion space.

LOO cross-validation across 14 rounds:

| Method | Avg Score | vs 2D IDW |
|--------|----------:|----------:|
| 1D Linear (V3) | 84.8 | -0.4 |
| **2D Kernel (V4)** | **87.3** | **+2.1** |
| 2D IDW | 85.2 | baseline |
| 1D Kernel | 87.0 | +1.9 |

### 5. Adaptive Bayesian Blending (V4, +1-3 pts on observed cells)

The KB gives a prior for each cell, but we also have direct stochastic observations. Combining them naively doesn't work because a single observation is just one Monte Carlo run.

Dirichlet-Multinomial update: KB distribution becomes the Dirichlet prior, observations update it. Concentration adapts: 1 observation gets a gentle update (conc=40), 5+ gets a strong one (conc=6).

### 6. Full-Grid Query Strategy (V4, eliminates 20% blind spots)

The greedy settlement-cover strategy left 17-23% of cells per seed with zero observations, so those cells were pure KB guesses.

Switched to systematic 3x3 grid tiling with 15x15 viewports. Covers 100% of the 40x40 map in 9 queries, with 1 query left for settlement redundancy.

| Strategy | 0-obs cells | 1-obs | 2+ obs |
|----------|----------:|------:|-------:|
| Greedy settlement-cover | 17-23% | 37-43% | 35-40% |
| **Full-grid + settlement** | **0%** | 77% | 23% |

### 7. API Rate Limiting

Added exponential-backoff retry (up to 5 attempts) on all endpoints. Previously a 429 mid-run would crash the solver and waste the remaining query budget.

---

## Architecture (V4)

### Pipeline

```
Static cell (mountain/ocean)?  ──────────────────────>  hardcoded prior
         │ no
Step 1: Estimate survival rate     MLE over ~10,000 cells
        from observations          Uses LINEAR interp in KB (with edge extrapolation)
         │
Step 2: Estimate expansion rate    Count plains->settlement transitions
         │
Step 3: Compute context key        terrain, coastal, dist-to-settlement, forest count
         │
Step 4: 2D Kernel interpolation    Gaussian kernel over all 19 historical rounds
        from expansion KB          bw_surv=0.07, bw_exp=0.10
         │
Step 5: Bayesian update            Dirichlet-Multinomial with adaptive concentration
        with per-cell observations  1 obs -> gentle (conc=40), 5+ obs -> strong (conc=6)
         │
Final:  Floor (0.01) + renorm     Double pass to ensure valid distribution
```

### Knowledge Bases

Two complementary KBs built from ground truth of all completed rounds:

**Survival-indexed** (`survival_indexed_kb.json`) -- 52 context keys x 19 rounds:
- Each key maps to sorted survival rates + corresponding class distributions
- Used for 1D interpolation (fallback) and MLE likelihood computation

**Expansion-indexed** (`expansion_indexed_kb.json`) -- 52 context keys x 19 rounds:
- Each key maps to (survival_rate, expansion_rate) pairs + distributions
- Used for primary 2D kernel interpolation during prediction

### Context Keys

| Type | Format | Example |
|------|--------|---------|
| Settlement | `settlement\|{coast}\|forest_{0-3}` | `settlement\|coastal\|forest_2` |
| Port | `port\|{coast}\|forest_{0-3}` | `port\|inland\|forest_1` |
| Plains | `plains\|near_sett_{0-10}\|{coast}` | `plains\|near_sett_3\|inland` |
| Forest | `forest\|near_sett_{0-10}[|coastal]` | `forest\|near_sett_1\|coastal` |

Fallback chain: decrease settlement distance, then drop coastal, then try alternative distance bins.

### Historical Survival Rates (19 rounds)

| Range | Rounds | Character |
|-------|--------|-----------|
| 0.00-0.07 | R3 (0.018), R19 (0.041), R10 (0.058), R8 (0.068) | Near-total collapse |
| 0.13-0.35 | R20 (0.130), R13 (0.226), R4 (0.235), R21 (0.246), R9 (0.275), R16 (0.294), R15 (0.328), R5 (0.330) | Low to moderate survival |
| 0.40-0.45 | R6 (0.415), R2 (0.415), R1 (0.419), R7 (0.423), R17 (0.454) | High survival |
| 0.50-0.63 | R11 (0.499), R14 (0.522), R12 (0.600), R18 (0.632) | Very high survival |

---

## Project Structure

```
AstarIsland/
├── config.py               # Auth token, constants, terrain-to-class mapping
├── client.py               # API client: all endpoints + retry/rate-limiting
├── model_v4.py             # V4 prediction engine: 2D kernel + Bayesian blending
├── model_v5.py             # Experimental V5: settlement health scoring (not used)
├── solve_v4.py             # Main solver: queries, MLE, prediction, submission
├── continue_r16.py         # Continuation script for mid-round re-runs
├── build_kb.py             # Build survival-indexed KB from GT data
├── build_expansion_kb.py   # Build 2D expansion-indexed KB from GT data
├── learn.py                # Post-round GT analysis + feature extraction
├── validate_v4.py          # Offline validation against saved GT
├── validate_v4_v5.py       # V4 vs V5 comparison harness
├── sweep_v5.py             # Parameter sweep for V5 configs
├── check_scores.py         # Per-round scores, ranks, and weights
├── check_status.py         # Active round detail inspection
├── find_rank.py            # Find team on leaderboard
├── inspect_kb.py           # Quick KB structure inspection
├── status.py               # One-liner: round status + budget + leaderboard
├── README.md
├── DOCS/                   # Challenge documentation
│   ├── API_ENDPOINTS.md
│   ├── OVERVIEW.md
│   ├── QUICKSTART.md
│   ├── SCORING.md
│   └── SIMULATION_MECHANICS.md
├── knowledge_base/
│   ├── survival_indexed_kb.json     # 52 keys x 19 rounds (1D)
│   ├── expansion_indexed_kb.json    # 52 keys x 19 rounds (2D)
│   └── cumulative_knowledge.json    # Legacy KB
├── round_data/
│   ├── gt_r{N}_seed{S}.npz         # Ground truth arrays (R1-R5)
│   └── observations_*.json          # Saved viewport observations per round
└── archive/                # Superseded code: V1/V2/V3 models, analysis scripts
```

### Key Files

| File | Purpose | Lines |
|------|---------|------:|
| `model_v4.py` | 2D kernel interp, MLE estimation, Bayesian blending, fallback chain | ~710 |
| `model_v5.py` | V5 experiment: settlement health, KB confidence weighting | ~300 |
| `learn.py` | GT download, feature extraction, training data export | 336 |
| `solve_v4.py` | Query planning (full-grid + settlement cover), predictions, submission | 293 |
| `validate_v4.py` | Offline validation: V3 oracle vs V4 variants on historical GT | 248 |
| `validate_v4_v5.py` | V4 vs V5 head-to-head comparison across all rounds | ~220 |
| `sweep_v5.py` | Parameter sweep: 13 V5 configs vs V4 baseline | ~220 |
| `build_kb.py` | Survival-indexed KB builder (from GT of all completed rounds) | 245 |
| `client.py` | REST client: simulate, submit, analysis, leaderboard (retry on 429) | 235 |
| `build_expansion_kb.py` | 2D KB builder (survival x expansion dimensions) | 186 |
| `config.py` | `BASE_URL`, `PROB_FLOOR=0.01`, `TERRAIN_TO_CLASS`, JWT token | 24 |

---

## Model Evolution

| Version | First Used | Best Score | Key Innovation |
|---------|-----------|:----------:|----------------|
| V1 | R1 | 27.3 | Heuristic priors |
| V2 | R2 | 80.1 | GT-calibrated lookup tables |
| V3 | R6 | 86.6 | Survival-indexed KB + MLE estimation |
| V4 | R15 | 89.0 | 2D kernel smoothing + expansion parameter + Bayesian blending |
| V5 | -- | -- | Experimental: settlement health scoring. Tested, no improvement over V4 |

### V3 to V4 changes

- Found second hidden parameter: expansion rate (probability plains cells become new settlements)
- Built 2D expansion-indexed KB (survival x expansion)
- Replaced linear interpolation with Gaussian kernel smoothing (bw_surv=0.07, bw_exp=0.10)
- Added adaptive Bayesian blending of KB prior with per-cell observations
- Extended MLE candidate range to [0.005, 0.85] with 340 candidates
- Linear interpolation for MLE now extrapolates beyond KB edges instead of clamping
- Kept linear interp for MLE only (kernel smoothing flattens the likelihood surface)
- Switched to full-grid query strategy: 9 systematic 15x15 viewports per seed, 100% coverage
- Added exponential-backoff retry on all API calls

### V5 experiment (abandoned)

Tried using settlement health stats (food, population, defense) from observations to create
soft labels for Bayesian updates. Also tested KB confidence weighting, death signal boosting,
and port transition modeling. Swept 13 configurations across 10 rounds with observations.
The best config (higher concentration [8,50]) gained +0.03 avg, which is noise. V4's
Bayesian concentration at [6,40] is already near optimal. The bottleneck is KB distribution
quality, not the update mechanics.

---

## Challenge Rules

- **Grid**: 40x40. 8 terrain codes, 6 prediction classes (Empty, Settlement, Port, Ruin, Forest, Mountain)
- **Seeds**: 5 per round. Same map + hidden params, different stochastic outcomes.
- **Queries**: 50 per round (shared across seeds). Each runs one 50-year sim, returns max 15x15 viewport.
- **Prediction**: WxHx6 probability tensor per seed. Must sum to 1.0 per cell, all values >= 0.01.
- **Scoring**: `score = 100 * exp(-3 * weighted_kl)`. Only dynamic cells contribute (weighted by entropy).
- **Leaderboard**: `max(round_score * round_weight)` -- only your best single weighted round matters.
- **Round weights** increase ~5% per round (R1=1.05, R19=2.53, R22=2.93).

---

## Lessons Learned

1. Survival rate is the master parameter. Estimate it well and the rest follows.
2. MLE over ~10k cells beats counting ~50 settlements. The error difference is 4-10x.
3. Don't average across rounds with different hidden parameters. Index by those parameters and interpolate.
4. Gaussian kernel smoothing uses all data points instead of just two neighbours. Worth +2.1 pts on average.
5. Persistent errors in a specific cell type usually mean there's a hidden parameter you haven't found yet.
6. Full grid coverage beats repeatedly querying the same settlements.
7. KL divergence explodes at p=0. Always floor at 0.01 and renormalize.
8. Ground truth is the average of hundreds of Monte Carlo runs. A single observation is one run. Use observations to estimate hidden parameters, not to set individual cell distributions.
9. Linear interpolation that clamps at KB edges is a silent killer. If the true parameter is beyond your training range, the MLE will just pick the edge value and you will not know it is wrong. Extrapolate instead.
10. More training rounds matter more than fancier update mechanics. V5 tried settlement health scoring, KB confidence weighting, death boosting, and port transition modeling. None of them moved the needle. The KB prior is already well calibrated; the limit is how many distinct (survival, expansion) regimes we have seen.
11. The leaderboard score is max(round_score x weight). A single great round on a late (high weight) round matters more than being consistently decent. Missing one bad round does not hurt you, but nailing one good round on R19+ can jump you 40 spots.

---

## Usage

```bash
pip install requests numpy

# Set JWT token in config.py (from app.ainm.no cookies, "access_token")

# Full round cycle:
python learn.py                # download GT from newly completed rounds
python build_kb.py             # rebuild survival-indexed KB
python build_expansion_kb.py   # rebuild 2D expansion KB
python solve_v4.py             # run queries + submit active round

# If solver crashes mid-run (rate limited), resume with:
python continue_r16.py         # loads saved obs, re-estimates, resubmits

# Utilities:
python status.py               # check round status / budget / leaderboard
python check_scores.py         # per-round scores and ranks
python validate_v4.py          # offline validation against historical GT
python inspect_kb.py           # inspect KB structure
```
