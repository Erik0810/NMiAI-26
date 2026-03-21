# NorgesGruppen Object Detection — Competition Solution

**Competition:** NM i AI (Norwegian AI Championships) — Grocery Object Detection  
**Competitor:** Erik Øverby, University of Oslo, Department of Informatics  
**Current best leaderboard score: 0.9095** (targeting 0.93+)

---

## Score Progression

| Version | Score  | What changed |
|---------|--------|--------------|
| Initial | 0.599  | YOLOv8m + EfficientNet-B0 classifier (buggy) |
| v1 fix  | 0.847  | Disabled EfficientNet — YOLO class IDs are already category_ids |
| v2      | 0.890  | YOLOv8x trained on GCP (1280px, 142 epochs) |
| v3      | 0.9095 | YOLOv8x trained at 1536px, all 248 images, 300 epochs |
| v5      | < v3   | 1792px model pulled too early (epoch 85/200) — undertrained |

---

## Scoring Formula

```
final_score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5
```

Both measured with COCO mAP@0.5. Key insight: **YOLO's class output IS the category_id (0-355) directly** — no separate classifier needed. The original EfficientNet-B0 stage was overwriting YOLO's trained class predictions with noisy cosine-similarity lookups, killing classification accuracy.

---

## Current Architecture

```
Shelf Image (avg ~3000x3100px)
    |
    v
Letterbox resize to 1536px or 1792px (preserves aspect ratio, grey pad)
    |
    |  x6 TTA passes
    |  scales: [1.0, 0.83, 1.17]
    |  flips:  [none, horizontal]
    v
YOLOv8x ONNX  (best.onnx, ~262 MB)
68M params, 257 GFLOPs
nc=356, trained on all 248 images
    |
    |  raw detections from all 6 passes
    v
Weighted Boxes Fusion (WBF)
iou_thr=0.55, skip_box_thr=0.07
merges overlapping boxes by weight
    |
    v
COCO predictions JSON
[{image_id, category_id, bbox, score}, ...]
```

No EfficientNet. No SAHI. No post-hoc classifier. YOLO's class head directly outputs category_ids.

---

## Dataset

| Item                  | Details                                            |
|-----------------------|----------------------------------------------------|
| Training images       | 248 shelf photos from Norwegian grocery stores      |
| Annotations           | 22,731 COCO-format bounding boxes                   |
| Categories            | 356 products (IDs 0-355)                            |
| Average image size    | ~3000x3100 pixels                                   |
| Store sections        | Egg, Frokost, Knekkebrod, Varmedrikker              |
| Train/Val split       | All 248 used for training (--no-val or val=full)    |

---

## Training Infrastructure

### GCP VMs
Both VMs: g2-standard-16, nvidia-l4 (24 GB VRAM), europe-west1-c, 200 GB disk  
Image: pytorch-2-7-cu128-ubuntu-2204-nvidia-570

| VM | Job | Status |
|----|-----|--------|
| objdet-train  | 3-fold k-fold ensemble (fold_0 resuming, fold_1+2 pending) | Running |
| objdet-train2 | 1792px fine-tune from ep299 checkpoint, 200 epochs | Running (~epoch 85+) |

### Training Scripts

| Script | Purpose |
|--------|---------|
| train_v3.py | Single-model training. Flags: --weights, --epochs, --imgsz, --batch, --no-val, --resume |
| train_kfold.py | K-fold cross-validation training, auto-resumes from last.pt |
| train_kfold_sequential.sh | Runs fold 0 then 1 then 2 sequentially, exports all as FP16 ONNX |

### Training Commands (on VM)
```bash
# Single model, fine-tune from checkpoint at higher resolution
python3 train_v3.py --weights best_v3_ep299.pt --epochs 200 --imgsz 1792 --batch 1 --no-val

# Resume after crash
python3 train_v3.py --resume

# K-fold ensemble (sequential)
bash train_kfold_sequential.sh
```

---

## Training History

### v1 — Local, YOLOv8m, 640px
First attempt. 11 epochs, mAP50 peaked at 0.469. GPU memory issues on local machine.

### v2 — Classifier bug fix (no retraining)
Discovered EfficientNet was overwriting YOLO class predictions with wrong category_ids.  
Set use_classifier = False. Score jumped from 0.599 to 0.847 without any retraining.

### v3 — GCP, YOLOv8x, 1280px (grocery_v3)
First GCP run on objdet-train (g2-standard-16, L4 GPU).  
YOLOv8x (68M params). Trained to epoch 142, mAP50=0.855. Score: 0.890

### v3-full — GCP, YOLOv8x, 1536px, all data (grocery_v3_full) — CURRENT BEST
Retrained from scratch at 1536px using --no-val (all 248 images in training).  
300 epochs. Best mAP50: 0.9748 at epoch 299. Added WBF replacing NMS for TTA merging.  
Score: 0.9095

| Best epochs | Precision | Recall | mAP50  | mAP50-95 |
|-------------|-----------|--------|--------|----------|
| Epoch 299   | 0.9456    | 0.9270 | 0.9748 | 0.7808   |
| Epoch 290   | 0.9514    | 0.9216 | 0.9742 | 0.7864   |
| Epoch 295   | 0.9426    | 0.9308 | 0.9743 | 0.7837   |

### v4 — SAHI tiling (abandoned)
Added SAHI tiling + 4-scale TTA. Degraded performance and caused grader timeouts.  
Fully removed. All SAHI code stripped from run.py and run_ensemble.py.

### v5 — GCP, YOLOv8x, 1792px (objdet-train2) — IN PROGRESS
Fine-tuning from grocery_v3_full best.pt (epoch 299) at 1792px, batch=1, 200 epochs.  
At epoch 85: mAP50 0.978 — already ahead of v3-full best (0.9748) but only 85/200 epochs.  
Submitted at epoch 85 and scored worse. Wait for epoch 200 before re-submitting.

### Parallel: 3-fold ensemble (objdet-train) — IN PROGRESS
train_kfold_sequential.sh training fold_0 then fold_1 then fold_2, 200 epochs each at 1280px.  
fold_0 crashed at epoch 54, resumed from last.pt. fold_1 + fold_2 pending.  
When complete: export FP16 ONNX (~130MB each, 3x ~390MB total, fits in 420MB limit).  
run.py auto-detects fold_*.onnx and ensembles with WBF automatically.

---

## Inference Pipeline (run.py)

### Key Constants
```python
YOLO_IMG_SIZE  = 1536        # fallback only — actual size read from ONNX model shape
CONF_THRESHOLD = 0.07        # low threshold to catch small/occluded products
WBF_IOU        = 0.55        # WBF merge threshold
TTA_SCALES     = [1.0, 0.83, 1.17]
TTA_FLIPS      = [False, True]
```

### TTA + Ensemble loop (per image)
```
for each model in [fold_0.onnx, fold_1.onnx, ...] (or best.onnx):
    for each scale in [1.0, 0.83, 1.17]:
        for each flip in [none, horizontal]:
            letterbox resize, run ONNX, decode boxes
            unflip coordinates if needed
            collect detections
WBF merge all detections -> final predictions
```

Model auto-detection: at startup run.py globs for fold_*.onnx. If found, loads all as ensemble.
Falls back to best.onnx. Same script handles both single-model and ensemble with no changes.

---

## Export and Packaging

```bash
# On VM after training completes (FP32, single model)
python3 export_and_package.py \
    --weights runs/detect/grocery_v3_full/weights/best.pt \
    --imgsz 1792 --name best.onnx
# produces submission.zip (~212 MB)

# FP16 fold models (needed to fit 3 models in 420 MB limit)
python3 export_and_package.py \
    --weights runs/detect/fold_0/weights/best.pt \
    --imgsz 1280 --half --name fold_0.onnx
```

```powershell
# Download finished submission from VM
& "C:\Users\erikk\...\gcloud.ps1" compute scp "objdet-train2:/home/erikk/objdet/submission.zip" D:\NMiAI\ObjectDetection\submission_v5.zip --zone=europe-west1-c
```

---

## Project Structure

```
ObjectDetection/
├── run.py                     # Submission entry point (inference + WBF + TTA)
├── run_ensemble.py            # Standalone ensemble runner (dev/debug only)
├── train_v3.py                # Single-model training script
├── train_kfold.py             # K-fold training script
├── train_kfold_sequential.sh  # Trains all 3 folds sequentially on VM
├── prepare_dataset.py         # COCO to YOLO format converter
├── export_and_package.py      # ONNX export + submission.zip packager
├── build_embeddings.py        # (legacy) EfficientNet embedding builder
├── visualize.py               # Detection visualizer
├── data.yaml                  # YOLO dataset config (nc=356)
│
├── best.onnx                  # Current best model (FP32, 1536px, ~262 MB)
├── best_v3_ep299.pt           # PyTorch checkpoint from objdet-train (ep299)
├── best_v4.pt                 # Early checkpoint from objdet-train2 (ep85, undertrained)
│
├── submission_v4_sahi.zip     # CURRENT BEST SUBMISSION (score 0.9095)
├── submission_v5.zip          # 1792px ep85 — scored worse, awaiting full run
│
├── dataset/                   # Raw COCO format dataset
├── dataset_yolo/              # Converted YOLO format (images/full, labels/full)
├── product_images/            # Product reference images (legacy, not used)
├── DOCS/                      # Competition docs
└── willow_application.txt     # Google Willow program access application
```

---

## Key Bug Fixes

| Bug | Fix | Score impact |
|-----|-----|-------------|
| EfficientNet overwriting YOLO class predictions | use_classifier = False | +0.248 |
| NMS discarding valid TTA detections | Replaced with WBF | ~+0.01 estimated |
| SAHI tiling causing grader timeouts | Fully removed | restored 0.9095 |
| fold_0 training crash at epoch 54 | last.pt auto-resume in train_kfold.py | unblocked ensemble |
| v5 submitted at only epoch 85/200 | Wait for epoch 200 | pending |

---

## Submission Constraints

| Constraint | Limit | Current |
|------------|-------|---------|
| Weight files (.pt, .onnx, .npy) | max 3 | 1 (best.onnx) or 3 (fold_*.onnx) |
| Total weight size | 420 MB | ~262 MB FP32 / ~390 MB 3x FP16 |
| Python files | max 10 | 1 (run.py) |
| Execution timeout | 300s | ~90s typical |
| Network access | none | ONNX Runtime only, all weights local |

Zip format: run.py at root + best.onnx (or fold_0/1/2.onnx).  
Output: COCO JSON list [{image_id, category_id, bbox, score}]. bbox: [x, y, w, h].

---

## What's Next

1. Wait for 1792px run to finish (~epoch 200 on objdet-train2)
   - Currently epoch 85+, mAP50 0.978 — already better than previous best at full 300 epochs
   - Export and submit when done

2. Wait for 3-fold ensemble (objdet-train)
   - fold_0 resuming, fold_1+2 pending
   - Export FP16: fold_0.onnx, fold_1.onnx, fold_2.onnx, package with run.py

3. Check progress:
```powershell
# 1792px run epoch count
& "C:\Users\erikk\...\gcloud.ps1" compute ssh objdet-train2 --zone=europe-west1-c --command="grep -v epoch /home/erikk/objdet/runs/detect/runs/detect/grocery_v3_full/results.csv | wc -l"

# k-fold progress
& "C:\Users\erikk\...\gcloud.ps1" compute ssh objdet-train --zone=europe-west1-c --command="ls runs/detect/ && tail -1 runs/detect/fold_0/results.csv 2>/dev/null"
```

4. Stop VMs when done:
```powershell
& "C:\Users\erikk\...\gcloud.ps1" compute instances stop objdet-train objdet-train2 --zone=europe-west1-c
```
