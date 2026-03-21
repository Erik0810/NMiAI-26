# Norwegian AI National Championships 2026

**Platform:** [app.ainm.no](https://app.ainm.no) &nbsp;|&nbsp; **Duration:** 4 days (March 2026)

---

## Overview

Four days of back-to-back AI engineering across computer vision, autonomous agents, and probabilistic simulation modelling — each scored live against a national leaderboard.

**Peak placement: 13th (Day 2).** Hovered around 30th for most of the competition before ultimately finishing at **XX**.

Key lessons across all three tasks:

- **Score the right thing.** Each scoring function had non-linear properties (mAP@0.5, KL divergence, API correctness) and understanding the curve early beat writing clean code.
- **Data beats model complexity.** The AstarIsland knowledge base grew from 1 to 13 rounds of ground-truth; each new round improved predictions more than any code change.
- **LLM function-calling is real engineering.** Wiring Gemini 2.5 Pro to a live REST API with structured self-correction required careful tool schema design, not just prompting.
- **Don't add a classifier where none is needed.** The biggest single score jump (+0.248) came from removing a post-hoc EfficientNet classifier that was overwriting YOLO's already-correct class predictions.
- **Deployment is part of the solution.** Two of three tasks required production serving (GCP Cloud Run, Docker sandbox) and the model alone was not enough.
- **Find the hidden parameter first.** In the stochastic simulator, estimating settlement survival rate via MLE over ~10,000 cells rather than ~50 settlements reduced estimation error by 4-10x.

---

## Task 1 — NorgesGruppen Grocery Object Detection

> Detect and classify grocery products on Norwegian store shelf images. Scored as `0.7 × detection mAP@0.5 + 0.3 × classification mAP@0.5` across 356 product categories.

### Architecture

![Object Detection Architecture](Static/object_detection_architecture.png)

### Example Detections

![Object Detection Example](Static/object_detection_example.jpg)

Local hardware wasn't enough for serious training, so I applied for and was granted access to **Google Cloud Platform** through the competition. I spun up two Linux VMs (g2-standard-16, each with an NVIDIA L4 GPU and 24 GB VRAM) and ran two training approaches in parallel: one going for a **3-fold ensemble** at 1280px, and one fine-tuning a single model at higher resolution (1536px then 1792px). After comparing results, the higher-resolution single model consistently outperformed the ensemble approach and became the final submission.

**Pipeline:** Single **YOLOv8x** model (68M params, trained at 1536px on all 248 shelf images, 22,731 annotations). YOLO's class head outputs category IDs 0-355 directly with no separate classifier needed. At inference: 6-pass TTA (3 scales × 2 flips) merged with Weighted Boxes Fusion. Best submitted score: **0.9095** (the leaderboard leader scored 0.9255, putting me at 98.3% of the top score).

**Stack:** Python 3.12 · YOLOv8x (Ultralytics) · ONNX · WBF · GCP g2-standard-16 · Docker

### Plateaus and fixes

| Plateau | Fix |
|---|---|
| EfficientNet classifier overwriting YOLO's correct class predictions (score 0.599) | Removed it entirely — YOLO class IDs are category IDs, score jumped to 0.847 |
| YOLOv8m at 640px capped mAP50 around 0.47 on local hardware | Moved to GCP, switched to YOLOv8x at 1536px, trained 300 epochs, mAP50 reached 0.9748 |
| NMS discarding valid TTA detections from multi-scale passes | Replaced NMS with Weighted Boxes Fusion for cleaner merging across scales and flips |

---

## Task 2 — Tripletex Accounting Agent

> Build an autonomous AI agent that receives natural-language accounting task prompts in any of 7 languages (Norwegian, Nynorsk, English, German, French, Spanish, Portuguese), interprets them, and executes the correct sequence of operations against the live [Tripletex](https://www.tripletex.no/) accounting API — covering 30 task types including creating employees, posting journal entries, reconciling bank transactions, managing invoices, and registering travel expenses.

### Architecture

![Tripletex Agent Architecture](Static/tripletex_agent_architecture.drawio.png)

**FastAPI** service deployed on **GCP Cloud Run** exposing a single `POST /solve` endpoint. Each submission gets a fresh Tripletex sandbox account. Gemini 2.5 Pro drives a multi-turn function-calling loop: plan the required API calls, execute them, then self-correct on errors by feeding the raw response body back into the model context. Scored field-by-field against expected values with an efficiency bonus.

**Stack:** Python 3.12 · FastAPI · Gemini 2.5 Pro · httpx · GCP Cloud Run · Docker

### Plateaus and fixes

| Plateau | Fix |
|---|---|
| Prompts arriving in 7 different languages confused intent parsing | Delegated all intent parsing to Gemini 2.5 Pro with a multilingual system prompt covering all 7 languages |
| Hard-coded field names caused schema mismatches against the live API | Verified every field against the Tripletex OpenAPI spec and embedded the constraints directly into function declarations |
| PUT requests failing silently due to missing `version` field | Enforced a GET-before-PUT pattern and fed raw 4xx error bodies back to the model for self-correction |

---

## Task 3 — Astar Island (Stochastic Civilisation Predictor)

> A 40x40 Norse settlement simulator runs 50-year stochastic simulations. Given 50 viewport queries per round, predict the full final-state probability distribution across 6 terrain classes for every cell. Leaderboard score is `round_score x round_weight`, where weights compound ~5% per round.

### Example

![Astar Island Layer Analysis](Static/astar_island_layer_analysis_example.png)

The simulation has two hidden parameters that change every round: the **settlement survival rate** (2% to 60% across rounds) and the **expansion rate** (how likely plains cells are to become new settlements). Getting these estimates right is the entire game. The model evolved through four versions, building a growing knowledge base of completed round ground-truth that improves every subsequent prediction automatically.

**Score progression:** V1 heuristic 27.3 raw / V2 GT-calibrated 80.1 raw / V3 MLE + survival KB 86.6 raw (141.1 weighted) / V4 Gaussian kernel + expansion KB + Bayesian blending **89.0 raw (185.9 weighted)**

**Final rank: #56 / 347 teams.** Gap to #1 was ~10.7 weighted points after Round 15.

**V4 pipeline:** estimate survival rate via MLE over ~10,000 observed cells, estimate expansion rate from plains transitions, look up each cell's context key in a 2D knowledge base indexed by both parameters, smooth the result with Gaussian kernel regression across all 15 historical rounds, then Bayesian-update the prior with any direct observations of that cell (Dirichlet-Multinomial, adaptive concentration). Every cell on the 40x40 grid gets at least one observation via a systematic 3x3 grid tiling strategy covering 100% of the map.

**Stack:** Python 3.12 · NumPy · JWT auth · Custom MLE optimiser · 2D JSON knowledge base (survival x expansion)

### Plateaus and fixes

| Plateau | Fix |
|---|---|
| V1/V2 static priors broke completely when survival rate shifted between rounds | Built a survival-indexed KB from completed-round ground-truth; interpolate between historical rounds at prediction time |
| Counting ~50 surviving settlements gave noisy survival estimates | MLE over all ~10,000 observed cells (plains, forest, ports, ruins all carry signal) reduced estimation error 4-10x |
| Near-settlement plains cells accounted for 39% of total prediction error even with good survival estimates | Discovered a second hidden parameter (expansion rate), built a 2D KB, and replaced linear interpolation with Gaussian kernel smoothing over all 15 historical rounds |

---

## Repository Structure

```
NMiAI/
├── ObjectDetection/     # Task 1 — YOLOv8x grocery detection (score 0.9095, 98.3% of leader)
├── AccountingAgent/     # Task 2 — Tripletex AI agent (FastAPI + Gemini 2.5 Pro, GCP Cloud Run)
├── AstarIsland/         # Task 3 — Norse civilisation predictor (MLE + survival-indexed KB)
└── Static/              # Architecture diagrams and example images
```

---

## Results Summary

| Task | Result |
|---|---|
| Object Detection | Score **0.9095** (98.3% of the leaderboard leader at 0.9255) |
| Tripletex Agent | Functional end-to-end; scored on task completion rate |
| Astar Island | **185.9 weighted** (89.0 raw) — V4, Round 15 — rank #56 / 347 |

**Peak leaderboard position: 13th (Day 2) · Final placement: XX**

---

*Built solo over 4 days. All models trained, deployed, and iterated on during the live competition window.*
