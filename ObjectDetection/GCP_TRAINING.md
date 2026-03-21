# GCP Training Guide — NorgesGruppen Object Detection

## Why GCP?

| What | Local (RTX 3060 Ti 8GB) | GCP (L4 24GB / A100 40GB) |
|------|-------------------------|---------------------------|
| Model | YOLOv8**m** (26M params) | YOLOv8**x** (68M params) |
| Resolution | 640px | **1280px** |
| Batch size | 4 | 8-16 (auto) |
| Training time | ~8 hrs (100 epochs) | ~3-6 hrs (300 epochs) |
| Expected score | ~0.60 | **~0.75-0.85** |

**Key insight**: The shelf images are up to 5712×4624px with ~92 small products per image. Training at 1280px instead of 640px hugely improves small object detection, which is the biggest factor for this competition.

## Quick Start

### 1. Create a GCP VM

```bash
# Option A: L4 GPU (cheapest, 24GB VRAM) — ~$1/hr
gcloud compute instances create objdet-train \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=200GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE

# Option B: A100 GPU (fastest, 40GB VRAM) — ~$3/hr  
gcloud compute instances create objdet-train \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-a100-80gb,count=1 \
  --boot-disk-size=200GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE
```

### 2. Upload data to the VM

```bash
# From your local machine:
gcloud compute scp NM_NGD_coco_dataset.zip objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp NM_NGD_product_images.zip objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp prepare_dataset.py objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp train_v3.py objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp build_embeddings.py objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp export_and_package.py objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp run.py objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp data.yaml objdet-train:~/objdet/ --zone=us-central1-a
gcloud compute scp gcp_setup.sh objdet-train:~/objdet/ --zone=us-central1-a
```

Or use a GCS bucket:
```bash
# Upload to bucket
gsutil cp NM_NGD_coco_dataset.zip gs://YOUR_BUCKET/
gsutil cp NM_NGD_product_images.zip gs://YOUR_BUCKET/

# On the VM, download from bucket
gsutil cp gs://YOUR_BUCKET/*.zip ~/objdet/
```

### 3. SSH into the VM

```bash
gcloud compute ssh objdet-train --zone=us-central1-a
```

### 4. Setup & extract data

```bash
cd ~/objdet

# Run setup (installs packages, pinned to sandbox versions)
chmod +x gcp_setup.sh
./gcp_setup.sh

# Extract datasets
mkdir -p dataset/train
cd dataset/train
unzip ~/objdet/NM_NGD_coco_dataset.zip
cd ~/objdet
mkdir -p product_images
cd product_images
unzip ~/objdet/NM_NGD_product_images.zip
cd ~/objdet

# Prepare YOLO format dataset
source .venv/bin/activate
python prepare_dataset.py
```

### 5. Train

```bash
# Use tmux so training survives SSH disconnect!
tmux new -s train

source .venv/bin/activate
cd ~/objdet

# Phase 1: Train with val split (monitor progress)
python train_v3.py --model yolov8x.pt --imgsz 1280 --epochs 300

# Check results
cat runs/detect/grocery_v3/results.csv | tail -5

# Phase 2: Final model — train on ALL images
python train_v3.py --model yolov8x.pt --imgsz 1280 --epochs 300 --no-val
```

**Detach tmux**: `Ctrl+B` then `D` (training continues in background)  
**Reattach**: `tmux attach -t train`

### 6. Export & package

```bash
# Build product embeddings (if not already done)
python build_embeddings.py

# Export best model to ONNX + build submission.zip
python export_and_package.py --imgsz 1280
```

### 7. Download submission

```bash
# On your local machine:
gcloud compute scp objdet-train:~/objdet/submission.zip . --zone=us-central1-a
```

### 8. Don't forget to stop the VM!

```bash
gcloud compute instances stop objdet-train --zone=us-central1-a
# Or delete it:
gcloud compute instances delete objdet-train --zone=us-central1-a
```

## What the improved training does

### Model: YOLOv8x vs YOLOv8m
- 68M params vs 26M → better feature extraction for 356 classes
- More capacity to distinguish similar-looking products
- Fits easily on 24GB VRAM

### Resolution: 1280px vs 640px
**This is the single biggest improvement.** Shelf images average ~3000×2000px with ~92 products each. Many products are tiny at 640px. At 1280px, small products retain enough detail to detect and classify.

### Augmentation (aggressive for 248 images)
- **Mosaic** (4 images combined): Effectively 4× more training data per batch
- **MixUp**: Blends images for regularization
- **Copy-paste**: Synthesizes new object placements
- **Random erasing**: Forces model to use context, not just one feature
- **Color jitter**: Handles different shelf lighting conditions

### Optimizer: AdamW vs SGD
- Better convergence on small datasets
- Weight decay prevents overfitting with only 248 images

### Run.py improvements  
- **Test-Time Augmentation (TTA)**: 3 scales × 2 flips = 6 inference passes. Catches products missed at any single scale. All merged with NMS.
- **ONNX with dynamic input size**: Supports any resolution at inference time
- **Lower confidence threshold** (0.10): Let the classifier handle borderline detections

## Expected timeline & costs

| Phase | Time | Cost (L4) |
|-------|------|-----------|
| Setup + data upload | 15 min | — |
| Phase 1 training (300 epochs w/ val) | 3-6 hrs | ~$4-6 |
| Phase 2 final training (300 epochs, all data) | 3-6 hrs | ~$4-6 |
| Export + package | 5 min | — |
| **Total** | **~7-12 hrs** | **~$8-12** |

## File checklist

Files to upload to GCP:
- [x] `NM_NGD_coco_dataset.zip` (854 MB)
- [x] `NM_NGD_product_images.zip` (60 MB)  
- [x] `prepare_dataset.py`
- [x] `train_v3.py`
- [x] `build_embeddings.py`
- [x] `export_and_package.py`
- [x] `run.py`
- [x] `data.yaml`
- [x] `gcp_setup.sh`

Files generated on GCP:
- `runs/detect/grocery_v3/weights/best.pt` (training output)
- `best.onnx` (ONNX export)
- `effnet_b0.onnx` (classifier export)
- `product_embeddings.npy` + `product_mapping.json`
- **`submission.zip`** ← download this

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `nvidia-smi` shows no GPU | VM needs GPU driver. Use `--image-family=pytorch-latest-gpu` |
| CUDA out of memory | Use `--batch -1` (auto) or reduce `--imgsz 960` |
| Training stuck/slow | Check `nvidia-smi` — if GPU util < 50%, increase `workers` |
| SSH disconnected | Use `tmux` — training continues in background |
| Permission denied on GCS | Run `gcloud auth login` on the VM |
