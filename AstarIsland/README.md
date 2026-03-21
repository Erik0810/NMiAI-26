# Astar Island Solver

Predict a stochastic Norse civilisation simulator's final state from limited viewport observations.

**Platform**: [app.ainm.no](https://app.ainm.no) | **API**: api.ainm.no/astar-island/  
**Team**: Pining for the fjords | **Rank**: #56 / 347 teams

---

## Current Status (2026-03-21)

| Round | Score | Weight | Weighted | Model | Survival | Expansion |
|-------|------:|-------:|---------:|-------|:--------:|:---------:|
| R1  | 27.3 | 1.050 | 28.7  | V1 (heuristic)     | 0.419 | 0.170 |
| R2  | 80.1 | 1.103 | 88.3  | V2 (GT-calibrated) | 0.415 | 0.205 |
| R6  | 78.2 | 1.340 | 104.7 | V3 (6-round KB)    | 0.415 | 0.264 |
| R7  | 64.2 | 1.407 | 90.4  | V3 (6-round KB)    | 0.423 | 0.147 |
| R10 | 86.6 | 1.629 | 141.1 | V3 (9-round KB)    | 0.058 | 0.009 |
| R14 | 77.4 | 1.980 | 153.3 | V3 (13-round KB)   | 0.405 | 0.265 |
| R15 | **89.0** | 2.079 | **185.0** | V4 (14-round KB) | 0.328 | 0.187 |
| R16 | -- | 2.183 | -- | V4 (15-round KB)    | est 0.331 | est 0.094 |

**Best**: R15 = 89.0 raw × 2.079 weight = **185.9 weighted** — **Rank #56 / 347**

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
        from observations          Uses LINEAR interp in KB (sharp likelihood)
         │
Step 2: Estimate expansion rate    Count plains→settlement transitions
         │
Step 3: Compute context key        terrain, coastal, dist-to-settlement, forest count
         │
Step 4: 2D Kernel interpolation    Gaussian kernel over all 15 historical rounds
        from expansion KB          bw_surv=0.07, bw_exp=0.10
         │
Step 5: Bayesian update            Dirichlet-Multinomial with adaptive concentration
        with per-cell observations  1 obs → gentle (conc=40), 5+ obs → strong (conc=6)
         │
Final:  Floor (0.01) + renorm     Double pass to ensure valid distribution
```

### Knowledge Bases

Two complementary KBs built from ground truth of all completed rounds:

**Survival-indexed** (`survival_indexed_kb.json`) — 52 context keys × 15 rounds:
- Each key maps to sorted survival rates + corresponding class distributions
- Used for 1D interpolation (fallback) and MLE likelihood computation

**Expansion-indexed** (`expansion_indexed_kb.json`) — 52 context keys × 15 rounds:
- Each key maps to (survival_rate, expansion_rate) pairs + distributions
- Used for primary 2D kernel interpolation during prediction

### Context Keys

| Type | Format | Example |
|------|--------|---------|
| Settlement | `settlement\|{coast}\|forest_{0-3}` | `settlement\|coastal\|forest_2` |
| Port | `port\|{coast}\|forest_{0-3}` | `port\|inland\|forest_1` |
| Plains | `plains\|near_sett_{0-10}\|{coast}` | `plains\|near_sett_3\|inland` |
| Forest | `forest\|near_sett_{0-10}[|coastal]` | `forest\|near_sett_1\|coastal` |

Fallback chain: decrease settlement distance → drop coastal → try alternative distance bins.

### Historical Survival Rates (15 rounds)

| Range | Rounds | Character |
|-------|--------|-----------|
| 0.00–0.10 | R3 (0.018), R10 (0.058), R8 (0.068) | Near-total collapse |
| 0.20–0.35 | R13 (0.226), R4 (0.235), R9 (0.275), R15 (0.328), R5 (0.330) | Low–moderate survival |
| 0.40–0.45 | R6 (0.415), R2 (0.415), R1 (0.419), R7 (0.423) | High survival |
| 0.50–0.60 | R11 (0.499), R14 (0.522), R12 (0.600) | Very high survival |

---

## Project Structure

```
AstarIsland/
├── config.py               # Auth token, constants, terrain-to-class mapping
├── client.py               # API client: all endpoints + retry/rate-limiting
├── model_v4.py             # V4 prediction engine: 2D kernel + Bayesian blending
├── solve_v4.py             # Main solver: queries, MLE, prediction, submission
├── continue_r16.py         # Continuation script for mid-round re-runs
├── build_kb.py             # Build survival-indexed KB from GT data
├── build_expansion_kb.py   # Build 2D expansion-indexed KB from GT data
├── learn.py                # Post-round GT analysis + feature extraction
├── validate_v4.py          # Offline validation against saved GT
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
│   ├── survival_indexed_kb.json     # 52 keys × 15 rounds (1D)
│   ├── expansion_indexed_kb.json    # 52 keys × 15 rounds (2D)
│   └── cumulative_knowledge.json    # Legacy KB
├── round_data/
│   ├── gt_r{N}_seed{S}.npz         # Ground truth arrays (R1–R5)
│   └── observations_*.json          # Saved viewport observations per round
└── archive/                # Superseded code: V1/V2/V3 models, analysis scripts
```

### Key Files

| File | Purpose | Lines |
|------|---------|------:|
| `model_v4.py` | 2D kernel interp, MLE estimation, Bayesian blending, fallback chain | 697 |
| `learn.py` | GT download, feature extraction, training data export | 335 |
| `solve_v4.py` | Query planning (full-grid + settlement cover), predictions, submission | 292 |
| `validate_v4.py` | Offline validation: V3 oracle vs V4 variants on historical GT | 247 |
| `build_kb.py` | Survival-indexed KB builder (from GT of all completed rounds) | 244 |
| `client.py` | REST client: simulate, submit, analysis, leaderboard (retry on 429) | 234 |
| `build_expansion_kb.py` | 2D KB builder (survival × expansion dimensions) | 186 |
| `config.py` | `BASE_URL`, `PROB_FLOOR=0.01`, `TERRAIN_TO_CLASS`, JWT token | 24 |

---

## Model Evolution

| Version | First Used | Best Score | Key Innovation |
|---------|-----------|:----------:|----------------|
| V1 | R1 | 27.3 | Heuristic priors |
| V2 | R2 | 80.1 | GT-calibrated lookup tables |
| V3 | R6 | 86.6 | Survival-indexed KB + MLE estimation |
| **V4** | **R15** | **89.0** | 2D kernel smoothing + expansion parameter + Bayesian blending |

### V3 to V4 changes

- Found second hidden parameter: expansion rate (probability plains cells become new settlements)
- Built 2D expansion-indexed KB (survival x expansion)
- Replaced linear interpolation with Gaussian kernel smoothing (bw_surv=0.07, bw_exp=0.10)
- Added adaptive Bayesian blending of KB prior with per-cell observations
- Extended MLE candidate range from [0.005, 0.50] to [0.005, 0.65]
- Kept linear interpolation for MLE only (kernel smoothing flattens the likelihood surface)
- Switched to full-grid query strategy: 9 systematic 15x15 viewports per seed, 100% coverage
- Added exponential-backoff retry on all API calls

---

## Challenge Rules

- **Grid**: 40×40. 8 terrain codes → **6 prediction classes** (Empty, Settlement, Port, Ruin, Forest, Mountain)
- **Seeds**: 5 per round. Same map + hidden params, different stochastic outcomes.
- **Queries**: 50 per round (shared across seeds). Each runs one 50-year sim, returns max 15×15 viewport.
- **Prediction**: W×H×6 probability tensor per seed. Must sum to 1.0 per cell, all values ≥ 0.01.
- **Scoring**: `score = 100 × exp(−3 × weighted_kl)`. Only dynamic cells contribute (weighted by entropy).
- **Leaderboard**: `max(round_score × round_weight)` — only your best single weighted round matters.
- **Round weights** increase ~5% per round (R1=1.05, R15=2.08, R16=2.18).

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

---

## Usage

```bash
pip install requests numpy

# Set JWT token in config.py (from app.ainm.no cookies → "access_token")

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
